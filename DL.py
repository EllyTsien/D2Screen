import paddle as pdl
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # 屏蔽RDKit的warning
import pickle as pkl
from argparse import ArgumentParser
import pandas as pd
import pgl
from pahelix.datasets.inmemory_dataset import InMemoryDataset
import random
from sklearn.model_selection import train_test_split
from paddle import optimizer 
from pprint import pprint
import paddle.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import os
import wandb
import paddle.nn as nn
import models.mlp as mlp


#
from preprocess import Input_ligand_preprocess
from GEM import GEM_smile_transfer
from evaluation_train import evaluation_train
from prediction import ModelTester

def collate_fn(data_batch):
    """
    Dataloader中的数据处理函数
    该函数输入一个batch的数据, 返回一个batch的(atom_bond_graph, bond_angle_graph, label)
    """
    atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_names = ["bond_dir", "bond_type", "is_in_ring"]
    bond_float_names = ["bond_length"]
    bond_angle_float_names = ["bond_angle"]
    atom_bond_graph_list = []   # 原子-键特征图
    bond_angle_graph_list = []  # 键-键角特征图
    label_list = []            # label
    for data_item in data_batch:
        graph = data_item['graph']
        ab_g = pgl.Graph(
                num_nodes=len(graph[atom_names[0]]),
                edges=graph['edges'],
                node_feat={name: graph[name].reshape([-1, 1]) for name in atom_names},
                edge_feat={
                    name: graph[name].reshape([-1, 1]) for name in bond_names + bond_float_names})
        ba_g = pgl.Graph(
                num_nodes=len(graph['edges']),
                edges=graph['BondAngleGraph_edges'],
                node_feat={},
                edge_feat={name: graph[name].reshape([-1, 1]) for name in bond_angle_float_names})
        atom_bond_graph_list.append(ab_g)
        bond_angle_graph_list.append(ba_g)
        label_list.append(data_item['label'])
    atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
    bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
    # TODO: reshape due to pgl limitations on the shape
    def _flat_shapes(d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])
    _flat_shapes(atom_bond_graph.node_feat)
    _flat_shapes(atom_bond_graph.edge_feat)
    _flat_shapes(bond_angle_graph.node_feat)
    _flat_shapes(bond_angle_graph.edge_feat)
    return atom_bond_graph, bond_angle_graph, np.array(label_list, dtype=np.float32)

def get_data_loader(mode, batch_size, index):
    if mode == 'train':
        # 训练模式下将train_data_list划分训练集和验证集，返回对应的DataLoader
        data_list = pkl.load(open('work/train_data_list.pkl', 'rb'))  # 读取data_list
        train_data_list, valid_data_list = train_test_split(data_list, test_size=0.2, random_state=42)
        print('train: {len(train_data_list)}, valid: {len(valid_data_list)}')
        train_dataset = InMemoryDataset(train_data_list)
        valid_dataset = InMemoryDataset(valid_data_list)
        train_data_loader = train_dataset.get_data_loader(
            batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_fn)
        valid_data_loader = valid_dataset.get_data_loader(
            batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_fn)
        return train_data_loader, valid_data_loader
    
    elif mode == 'test':
        # 推理模式下直接读取test_data_list, 返回test_data_loader
        file_path = f'datasets/ZINC20_processed/{index}_ZINC20_data_list.pkl'
        data_list = pkl.load(open(file_path, 'rb'))
        if len(data_list) == 0:
            raise ValueError("Dataset is empty")
        test_dataset = InMemoryDataset(data_list)
        test_data_loader = test_dataset.get_data_loader(
            batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn)
        return test_data_loader
    
    
def trial(model_version, model, batch_size, criterion, scheduler, opt):
    # 创建dataloader
    train_data_loader, valid_data_loader = get_data_loader(mode='train', batch_size=batch_size, index=0)   
    current_best_metric = -1e10
    max_bearable_epoch = 50    # 设置早停的轮数为50，若连续50轮内验证集的评价指标没有提升，则停止训练
    current_best_epoch = 0
    train_metric_list = []     # 记录训练过程中各指标的变化情况
    valid_metric_list = []
    for epoch in range(800):   # 设置最多训练800轮
        model.train()
        for (atom_bond_graph, bond_angle_graph, label_true_batch) in train_data_loader:
            label_predict_batch = model(atom_bond_graph, bond_angle_graph)
            label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
            loss = criterion(label_predict_batch, label_true_batch)
            loss.backward()   # 反向传播
            opt.step()   # 更新参数
            opt.clear_grad()
        scheduler.step()   # 更新学习率
        # 评估模型在训练集、验证集的表现
        evaluator = evaluation_train(model, train_data_loader, valid_data_loader)
        metric_train = evaluator.evaluate(train_data_loader)
        metric_valid = evaluator.evaluate(valid_data_loader)
        train_metric_list.append(metric_train)
        valid_metric_list.append(metric_valid)
        score = round((metric_valid['ap'] + metric_valid['auc']) / 2, 4)
        if score > current_best_metric:
            # 保存score最大时的模型权重
            current_best_metric = score
            current_best_epoch = epoch
            pdl.save(model.state_dict(), "weight/" + model_version + ".pkl")
        print("=========================================================")
        print("Epoch", epoch)
        pprint(("Train", metric_train))
        pprint(("Validate", metric_valid))
        print('current_best_epoch', current_best_epoch, 'current_best_metric', current_best_metric)
        if epoch > current_best_epoch + max_bearable_epoch:
            break
    # evaluator.plot_metrics([train_metric_list], [valid_metric_list])
    return train_metric_list, valid_metric_list        

# 将测试集的预测结果保存为result.csv
def test(model_version, index):
    test_data_loader = get_data_loader(mode='test', batch_size=1024, index=index)
    # model = ADMET()
    model.set_state_dict(pdl.load("weight/" + model_version + ".pkl"))   # 导入训练好的的模型权重
    model.eval()
    all_result = []
    for (atom_bond_graph, bond_angle_graph, label_true_batch) in test_data_loader:
        label_predict_batch = model(atom_bond_graph, bond_angle_graph)
        label_predict_batch = F.softmax(label_predict_batch)
        result = label_predict_batch[:, 1].cpu().numpy().reshape(-1).tolist()
        all_result.extend(result)
    nolabel_file_path = f'datasets/ZINC20_processed/{index}_ZINC20_nolabel.csv'
    df = pd.read_csv(nolabel_file_path)
    # df = pd.read_csv('datasets/ZINC20_processed/test_nolabel.csv')
    df['pred'] = all_result
    result_file_path = 'datasets/DL_pred/result.csv'
    # 检查文件是否存在
    if os.path.exists(result_file_path):
        # 如果文件存在，则追加数据
        df.to_csv(result_file_path, mode='a', header=False, index=False)
    else:
        # 如果文件不存在，则创建文件并写入数据
        df.to_csv(result_file_path, index=False)


def sort_and_filter_csv(file_path, threshold, output_file_path):
    """
    读取CSV文件，根据“pred”这一列的值对数据进行排序，并筛选出“pred”值大于threshold的行，生成新的CSV文件
    参数:
    file_path (str): 输入的CSV文件路径
    threshold (float): 过滤的阈值
    output_file_path (str): 输出的CSV文件路径
    返回:
    None
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 检查是否存在“pred”列
        if 'pred' not in df.columns:
            raise ValueError("CSV文件中不存在'pred'这一列")
        # 根据“pred”这一列的值进行排序
        sorted_df = df.sort_values(by='pred')
        # 筛选出“pred”值大于threshold的行
        filtered_df = sorted_df[sorted_df['pred'] > threshold]
        # 将结果保存到新的CSV文件
        filtered_df.to_csv(output_file_path, index=False)
        print(f"生成的文件已保存到: {output_file_path}")
    except FileNotFoundError:
        print(f"文件路径错误或文件不存在: {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")
            
def plot(train, valid, metric, filename):
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train, color="blue", label='Training %s' %(metric))
    plt.plot(epochs, valid, color="orange", label='Validation %s' %(metric))
    plt.title('Training and validation %s' %(metric))
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory



def main(args):
    global model
    if args.model == "ADMET":
        model = mlp.ADMET() 
    else:
        raise NotImplementedError("Unknown model")
    
    # Initialize wandb with project name and config
    wandb.init(project=args.project_name, config={
        "seed": args.seed,
        "model": args.model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "threshold": args.threshold,
        "model_details": str(model)
    })
    
    config = wandb.config  # Use wandb config for consistency

    np.random.seed(config.seed)
    # torch.random.manual_seed(config.seed)
    
    if config.model is None:
        print("No model specified. Use '--model <model_name>'")
        exit(-1)

    if config.dataset is None:
        print("No dataset specified. Use '--dataset <dataset_name>'")
        exit(-1)
    else:
        input_ligands_path = 'datasets/' + config.dataset
        processed_input_path = 'datasets/train_preprocessed.csv'
    
    processor = Input_ligand_preprocess(input_ligands_path)
    processor.preprocess() 
    processed_input_csv = pd.read_csv(processed_input_path)
    SMILES_Transfer = SMILES_Transfer(processed_input_csv)
    SMILES_Transfer.run()
    
    
    if config.model == "ADMET":
        model = mlp.ADMET() 
    else:
        raise NotImplementedError("Unknown model")
    
    # Log model architecture
    wandb.config.update({"model_details": str(model)}, allow_val_change=True)
    # Fix random seed
    SEED = 1024
    np.random.seed(SEED)
    random.seed(SEED)
    # If using PyTorch
    # torch.manual_seed(SEED)
    
    batch_size = config.batch_size
    criterion = nn.CrossEntropyLoss() 
    scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=config.learning_rate, T_max=15)
    opt = optimizer.Adam(scheduler, parameters=model.parameters(), weight_decay=1e-5)

    # Train and validate the model
    metric_train_list, metric_valid_list = trial(model_version='1', model=model, batch_size=batch_size, criterion=criterion, scheduler=scheduler, opt=opt)
    
    # Convert to DataFrame for plotting
    metric_train = pd.DataFrame(metric_train_list)
    metric_valid = pd.DataFrame(metric_valid_list)

    # Log the training and validation metrics in wandb
    for metric in ['accuracy', 'ap', 'auc', 'f1', 'precision', 'recall']:
        wandb.log({
            f"train_{metric}": metric_train[metric].iloc[-1],  # Log the last value for simplicity
            f"valid_{metric}": metric_valid[metric].iloc[-1]
        })
    
    # Save performance plots and log them to wandb
    for metric in ['accuracy', 'ap', 'auc', 'f1', 'precision', 'recall']:
        filename = f'performance/{metric}_plot.png'
        plot(metric_train[metric], metric_valid[metric], metric=metric, filename=filename)
        wandb.log({f"{metric}_plot": wandb.Image(filename)})
    
    print('Evaluation plot saved!')
    '''
    # Test and log results
    for index in range(1, 23): 
        test(model_version='1', index=index)
    '''
    # Sort, filter and log the final result
    sort_and_filter_csv("datasets/DL_pred/result.csv", config.threshold, "datasets/DL_pred/top.csv")

    # Finish the run
    wandb.finish()


    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default='ADMET', type=str, help='Type of model to train (required)')
    parser.add_argument('--project_name', default='your_project_name', type=str, help='Name your project on the wandb website')
    parser.add_argument('--dataset', default='input.csv', type=str, help='Choose dataset (required)')
    parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size (default: 32)")
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for predict value (defalt 0.9)")
    args = parser.parse_args()
    
    main(args)