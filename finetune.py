import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')
import paddle as pdl
from paddle import optimizer 
import numpy as np
import pandas as pd
import json
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # 屏蔽RDKit的warning
import pickle as pkl
from argparse import ArgumentParser
from pahelix.datasets.inmemory_dataset import InMemoryDataset
import random
from sklearn.model_selection import train_test_split

from pprint import pprint
import paddle.nn as nn
import paddle.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import os
import wandb
# wandb.init(mode='disabled')

#
from finetunemodels import mlp
from preprocess import Input_ligand_preprocess,  SMILES_Transfer
from evaluation_train import evaluation_train
from prediction import ModelTester
from dataloader import collate_fn, get_data_loader, sort_and_filter_csv
from pahelix.model_zoo.gem_model import GeoGNNModel

# def trial(model_version, model, batch_size, criterion, scheduler, opt):
def run_finetune(params):
    finetune_model_config_path, lr, head_lr, dropout_rate, ft_time, batch_size, project_name, finetune_dataset, model_version = params
    seed = random.randint(0, 1000000)  # 可以根据需要调整范围
    finetune_model_config =json.load(open(finetune_model_config_path, 'r'))
    if not dropout_rate is None:
        finetune_model_config['dropout_rate'] = dropout_rate
    finetune_model_config['task_type'] = 'class'
    finetune_model_config['num_tasks'] = 1
    
    # Initialize wandb with project name and config
    wandb.init(project=project_name, config={
        "seed": seed,
        "finetunemodel": finetune_model_config,
        "dataset": finetune_dataset, 
        "batch_size": batch_size,
        "learning_rate": float(lr),
        "head_lr": float(head_lr),
        "finetune time": float(ft_time),
        "dropout rate": float(dropout_rate),
        "model_details": str(finetune_model_config)
    })
    config = wandb.config  # Use wandb config for consistency
    # Log model architecture
    wandb.config.update({"model_details": str(finetune_model_config)}, allow_val_change=True)
    np.random.seed(config.seed)
    random.seed(config.seed)

    compound_encoder_config = json.load(open('GEM/model_configs/geognn_l8.json', 'r')) 
    compound_encoder = GeoGNNModel(compound_encoder_config)
    '''
    # 这里显式传入MLP构造函数所需的所有参数
    finetune_model = mlp.MLP(
        layer_num=finetune_model_config['layer_num'],
        in_size=compound_encoder.graph_dim,
        hidden_size=finetune_model_config['hidden_size'],
        out_size=finetune_model_config['num_tasks'],
        act=finetune_model_config['act'],
        dropout_rate=finetune_model_config.get('dropout_rate', None)  # 使用从配置中获取的dropout_rate
    )
    '''
    finetune_model = mlp.MLP4()
    criterion = nn.CrossEntropyLoss() 
    scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=config.learning_rate, T_max=15)
    opt = optimizer.Adam(scheduler, parameters=finetune_model.parameters(), weight_decay=1e-5)

    # 创建dataloader
    train_data_loader, valid_data_loader = get_data_loader(mode='train', batch_size=batch_size, index=0)   
    current_best_metric = -1e10
    max_bearable_epoch = 50    # 设置早停的轮数为50，若连续50轮内验证集的评价指标没有提升，则停止训练
    current_best_epoch = 0
    train_metric_list = []     # 记录训练过程中各指标的变化情况
    valid_metric_list = []
    for epoch in range(800):   # 设置最多训练800轮
        finetune_model.train()
        for (atom_bond_graph, bond_angle_graph, label_true_batch) in train_data_loader:
            label_predict_batch = finetune_model(atom_bond_graph, bond_angle_graph)
            label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
            loss = criterion(label_predict_batch, label_true_batch)
            loss.backward()   # 反向传播
            opt.step()   # 更新参数
            opt.clear_grad()
        scheduler.step()   # 更新学习率
        # 评估模型在训练集、验证集的表现
        evaluator = evaluation_train(finetune_model, train_data_loader, valid_data_loader)
        metric_train = evaluator.evaluate(train_data_loader)
        metric_valid = evaluator.evaluate(valid_data_loader)
        train_metric_list.append(metric_train)
        valid_metric_list.append(metric_valid)
        score = round((metric_valid['ap'] + metric_valid['auc']) / 2, 4)
        if score > current_best_metric:
            # 保存score最大时的模型权重
            current_best_metric = score
            current_best_epoch = epoch
            pdl.save(finetune_model.state_dict(), "weight/" + model_version + ".pkl")
        print("=========================================================")
        print("Epoch", epoch)
        pprint(("Train", metric_train))
        pprint(("Validate", metric_valid))
        print('current_best_epoch', current_best_epoch, 'current_best_metric', current_best_metric)
        for metric in ['accuracy', 'ap', 'auc', 'f1', 'precision', 'recall']:
            wandb.log({
                f"train_{metric}": metric_train[metric].tolist(),  # Log the last value for simplicity
                f"valid_{metric}": metric_valid[metric].tolist()
            })
        if epoch > current_best_epoch + max_bearable_epoch:
            break
    # evaluator.plot_metrics([train_metric_list], [valid_metric_list])
    # Finish the run
    wandb.finish()
    return train_metric_list, valid_metric_list        

# 将测试集的预测结果保存为result.csv
def test(model_version, index):
    test_data_loader = get_data_loader(mode='test', batch_size=1024, index=index)
    # model = ADMET()
    finetune_model.set_state_dict(pdl.load("weight/" + model_version + ".pkl"))   # 导入训练好的的模型权重
    finetune_model.eval()
    all_result = []
    for (atom_bond_graph, bond_angle_graph, label_true_batch) in test_data_loader:
        label_predict_batch = finetune_model(atom_bond_graph, bond_angle_graph)
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
    
    # Train and validate the model
    # metric_train_list, metric_valid_list = trial(model_version='1', model=finetune_model, batch_size=batch_size, criterion=criterion, scheduler=scheduler, opt=opt)
    
    # Convert to DataFrame for plotting
    # metric_train = pd.DataFrame(metric_train_list)
    # metric_valid = pd.DataFrame(metric_valid_list)
# 
    # print(type(metric_train['accuracy']))
    # print(metric_train['accuracy'])
    # 
    # # Save performance plots and log them to wandb
    # for metric in ['accuracy', 'ap', 'auc', 'f1', 'precision', 'recall']:
    #     filename = f'performance/{metric}_plot.png'
    #     plot(metric_train[metric], metric_valid[metric], metric=metric, filename=filename)
    #     wandb.log({f"{metric}_plot": wandb.Image(filename)})
    # 
    # print('Evaluation plot saved!')

    # Test and log results
    for index in range(1, 23): 
        test(model_version='1', index=index)
    # Sort, filter and log the final result
    sort_and_filter_csv("datasets/DL_pred/result.csv", args.threshold, "datasets/DL_pred/top.csv")

    # Finish the run
    wandb.finish()


    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--finetunemodel', default='mlp4', type=str, help='Type of model to train (required)')
    parser.add_argument('--project_name', default='your_project_name', type=str, help='Name your project on the wandb website')
    parser.add_argument('--dataset', default='input.csv', type=str, help='Choose dataset (required)')
    parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size (default: 32)")
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for predict value (defalt 0.9)")
    args = parser.parse_args()
    
    run_finetune(args)