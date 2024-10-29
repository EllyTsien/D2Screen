import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')
import paddle as pdl
from paddle import optimizer 
import numpy as np
import pandas as pd
# from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # 屏蔽RDKit的warning
import pickle as pkl
from argparse import ArgumentParser
# import pgl
from pahelix.datasets.inmemory_dataset import InMemoryDataset
import random
from sklearn.model_selection import train_test_split

from pprint import pprint
import paddle.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import os
import wandb
import paddle.nn as nn

#
from finetunemodels import mlp
from preprocess import Input_ligand_preprocess,  SMILES_Transfer
from evaluation_train import evaluation_train
from prediction import ModelTester
from dataloader import collate_fn, get_data_loader, sort_and_filter_csv


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
    # passing parameters
    global finetune_model
    if args.finetunemodel == "mlp4":
        finetune_model = mlp.MLP4()
    elif args.finetunemodel == "mlp6":
        finetune_model = mlp.MLP6()
    else:
        raise NotImplementedError("No model specified. Use '--finetunemodel <model_name>'")
    
    if args.dataset is None:
        print("No dataset specified. Use '--dataset <dataset_name>'")
        exit(-1)
    else:
        input_ligands_path = 'datasets/' + args.dataset
        processed_input_path = 'datasets/train_preprocessed.csv'

    # process train dataset
    processor = Input_ligand_preprocess(input_ligands_path)
    processor.preprocess() 
    processed_input_csv = pd.read_csv(processed_input_path)
    SMILES_transfer = SMILES_Transfer(processed_input_csv)
    SMILES_transfer.run()
    
    # Initialize wandb with project name and config
    wandb.init(project=args.project_name, config={
        "seed": args.seed,
        "finetunemodel": args.finetunemodel,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "threshold": args.threshold,
        "model_details": str(finetune_model)
    })
    config = wandb.config  # Use wandb config for consistency
    # Log model architecture
    wandb.config.update({"model_details": str(finetune_model)}, allow_val_change=True)
    np.random.seed(config.seed)
    random.seed(config.seed)
    batch_size = config.batch_size

    criterion = nn.CrossEntropyLoss() 
    scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=config.learning_rate, T_max=15)
    opt = optimizer.Adam(scheduler, parameters=finetune_model.parameters(), weight_decay=1e-5)

    # Train and validate the model
    metric_train_list, metric_valid_list = trial(model_version='1', model=finetune_model, batch_size=batch_size, criterion=criterion, scheduler=scheduler, opt=opt)
    
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
    parser.add_argument('--finetunemodel', default='mlp4', type=str, help='Type of model to train (required)')
    parser.add_argument('--project_name', default='your_project_name', type=str, help='Name your project on the wandb website')
    parser.add_argument('--dataset', default='input.csv', type=str, help='Choose dataset (required)')
    parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size (default: 32)")
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for predict value (defalt 0.9)")
    args = parser.parse_args()
    
    main(args)