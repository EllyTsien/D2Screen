import paddle as pdl
from paddle import optimizer 
import numpy as np
import json
from rdkit import RDLogger
from pahelix.datasets.inmemory_dataset import InMemoryDataset
import random
import pandas as pd
from pprint import pprint
import paddle.nn as nn
import paddle.nn.functional as F
import os
import wandb
import glob
import shutil

from finetunemodels import mlp
from preprocess import Input_ligand_preprocess,  SMILES_Transfer
from evaluation_train import evaluation_train
from prediction import ModelTester
from dataloader import collate_fn, get_data_loader, sort_and_filter_csv
from pahelix.model_zoo.gem_model import GeoGNNModel


def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res


# def trial(model_version, model, batch_size, criterion, scheduler, opt):
def run_finetune(params):
    finetune_model_layer, lr, head_lr, dropout_rate, ft_time, batch_size, project_name, finetune_dataset, model_version = params
    seed =  42 
    # finetune_model_layer =json.load(open(finetune_model_layer, 'r'))
    if finetune_model_layer=='mlp4':
        finetune_model = mlp.MLP4()
    elif finetune_model_layer=='mlp6':
        finetune_model = mlp.MLP6()
    
    # Initialize wandb with project name and config
    run = wandb.init(project=project_name, config={
        "seed": seed,
        "finetunemodel": finetune_model_layer,
        "dataset": finetune_dataset, 
        "batch_size": batch_size,
        "learning_rate": float(lr),
        "head_lr": float(head_lr),
        "finetune time": float(ft_time),
        "dropout rate": float(dropout_rate),
        "model_details": str(finetune_model_layer)
    })

    # add 3in1 tables
    metric_logs = {
        metric: []
        for metric in ['accuracy', 'ap', 'auc', 'f1', 'precision', 'recall']
    }

    config = wandb.config  # Use wandb config for consistency
    # Log model architecture
    wandb.config.update({"model_details": str(finetune_model_layer)}, allow_val_change=True)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    #model construction
    compound_encoder_config = json.load(open('GEM/model_configs/geognn_l8.json', 'r')) 
    compound_encoder = GeoGNNModel(compound_encoder_config)
    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(finetune_model.parameters(), encoder_params)
    criterion = nn.CrossEntropyLoss() 
    encoder_scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=config.learning_rate, T_max=15)
    head_scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=config.head_lr, T_max=15)
    encoder_opt = optimizer.Adam(encoder_scheduler, parameters=encoder_params, weight_decay=1e-5)
    head_opt = optimizer.Adam(head_scheduler, parameters=head_params, weight_decay=1e-5)

    # 创建dataloader
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(mode='train', batch_size=batch_size, index=0)
    current_best_metric = -1e10
    max_bearable_epoch = 10    # 设置早停的轮数为50，若连续50轮内验证集的评价指标没有提升，则停止训练
    current_best_epoch = 0
    train_metric_list = []     # 记录训练过程中各指标的变化情况
    valid_metric_list = []
    test_metric_list = []
    for epoch in range(800):   # 设置最多训练800轮
        finetune_model.train()
        for (atom_bond_graph, bond_angle_graph, label_true_batch) in train_data_loader:
            label_predict_batch = finetune_model(atom_bond_graph, bond_angle_graph)
            label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
            loss = criterion(label_predict_batch, label_true_batch)
            loss.backward()   # 反向传播
            encoder_opt.step()   # 更新参数
            head_opt.step()   # 更新参数
            encoder_opt.clear_grad()
            head_opt.clear_grad()
        encoder_scheduler.step()   # 更新学习率
        head_scheduler.step() # 更新学习率
        # 评估模型在训练集、验证集的表现
        evaluator = evaluation_train(finetune_model, train_data_loader, valid_data_loader, test_data_loader)
        metric_train = evaluator.evaluate(train_data_loader)
        metric_valid = evaluator.evaluate(valid_data_loader)
        metric_test = evaluator.evaluate(test_data_loader)
        train_metric_list.append(metric_train)
        valid_metric_list.append(metric_valid)
        test_metric_list.append(metric_test)
        score = round((metric_valid['ap'] + metric_valid['auc']) / 2, 4)
        if score > current_best_metric:
            # 保存score最大时的模型权重
            current_best_metric = score
            current_best_epoch = epoch

            os.makedirs("bestweights" + "_" + project_name, exist_ok=True)
            os.makedirs("bestmodels" + "_" + project_name, exist_ok=True)
            run_id = run.name
            model_path = os.path.join("bestweights" + "_" + project_name, f"{run_id}.pkl")
            json_path = os.path.join("bestmodels" + "_" + project_name, f"{run_id}.json")
            pdl.save(finetune_model.state_dict(), model_path)

            # save best model config to .json
            best_model_info = {
                "score": score,
                "finetune_model_layer": finetune_model_layer,
                "learning_rate": lr,
                "head_learning_rate": head_lr,
                "dropout_rate": dropout_rate,
                "finetune_time": ft_time,
                "batch_size": batch_size,
                "project_name": project_name,
                "finetune_dataset": finetune_dataset,
                "model_version": model_version,
                "saved_model": model_path, 
                "run_id": run_id             
            }
            # 确保目标文件夹存在
            os.makedirs("finetunemodels", exist_ok=True)
            # 将配置信息保存为JSON文件
            with open(json_path, "w") as json_file:
                json.dump(best_model_info, json_file, indent=4)

        print("=========================================================")
        print("Epoch", epoch)
        pprint(("Train", metric_train))
        pprint(("Validate", metric_valid))
        pprint(("Test", metric_test))
        print('current_best_epoch', current_best_epoch, 'current_best_metric', current_best_metric)
        
        for metric in ['accuracy', 'ap', 'auc', 'f1', 'precision', 'recall']:
            # ***
            metric_logs[metric].append((epoch, "train", metric_train[metric]))
            metric_logs[metric].append((epoch, "validation", metric_valid[metric]))
            metric_logs[metric].append((epoch, "test", metric_test[metric]))

            wandb.log({
                f"train_{metric}": metric_train[metric],  # Log the last value for simplicity
                f"validation_{metric}": metric_valid[metric],
                f"test_{metric}": metric_test[metric]
            })
 
        if epoch > current_best_epoch + max_bearable_epoch:
            break

        pdl.device.cuda.empty_cache()

    # summarize data ***
    for metric, records in metric_logs.items():
        # 构造 DataFrame
        df = pd.DataFrame(records, columns=["epoch", "dataset", metric])
        # 确保 epoch 按升序排列
        df = df.sort_values("epoch")
        # 获取所有唯一的组名
        groups = df["dataset"].unique().tolist()
        xs = df["epoch"].unique().tolist()
        
        ys = []
        keys = []
        for group in groups:
            # 针对每个 group，取出该组的所有数据，并按照 epoch 排序
            group_df = df[df["dataset"] == group].sort_values("epoch")
            # 提取 y 值
            y_vals = group_df[metric].tolist()
            ys.append(y_vals)
            keys.append(group)
        
        # 调用 wandb.plot.line_series

        rows = []
        for key_index, key in enumerate(keys):
            for i, step in enumerate(xs):
                rows.append({
                    "step": step,
                    "lineVal": ys[key_index][i],
                    "lineKey": key
                })

        # 构建 DataFrame 并转为 wandb.Table
        df = pd.DataFrame(rows)
        table = wandb.Table(dataframe=df)

        # 使用自定义预设“3linesIn1Graph_color”生成图表
        chart = wandb.plot_table(
            vega_spec_name="121090453-the-chinese-university-of-hong-kong-shenzhen/3linesin1graph_color",
            data_table=table,
            fields={
                "step": "step", 
                "lineVal": "lineVal",
                "lineKey": "lineKey",     
                "color": "lineKey" 
            },
            string_fields={
                "title": f"{metric.upper()} over Epochs",  # 图表标题
                "xname": "Epochs"                          # x轴名称，可根据预设中 Vega 规范对应字段命名
            }
        )

        # 记录图表到 wandb
        wandb.log({f"{metric}_comparison_plot": chart})

    wandb.finish()
    return train_metric_list, valid_metric_list, test_metric_list        

def select_best_model(model_version, project_name):
    best_json_dir = "bestmodels_" + project_name
    best_weight_dir = "bestweights_" + project_name
    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(best_json_dir, "*.json"))
    best_score = -float("inf")
    best_info = None

    # 遍历所有 JSON 文件，挑选出 score 最高的
    for json_file in json_files:
        with open(json_file, "r") as f:
            info = json.load(f)
        # 要求各线程在保存 JSON 时将 score 存入其中，例如 info["score"] = score
        score = info.get("score", None)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_info = info

    if best_info is not None:
        # 构造原始保存路径
        best_json_target = os.path.join(best_json_dir, "best.json")
        best_model_target = os.path.join(best_weight_dir, f"{model_version}.pkl")
        
        # 源文件路径可以从 best_info 中获得
        best_run_id = best_info.get("run_id")
        # 假设保存时 JSON 的命名格式为 f"best_{run_id}.json"
        best_json_source = os.path.join(best_json_dir, f"{best_run_id}.json")
        best_model_source = best_info.get("saved_model")  # 例如保存在 bestweights 文件夹下

        # 复制 JSON 文件
        shutil.copy2(best_json_source, best_json_target)
        # 复制模型权重文件
        shutil.copy2(best_model_source, best_model_target)
        print(f"Selected best model with score: {best_score}")
    else:
        print("No valid best model found.")


# 将测试集的预测结果保存为result.csv
def test(model_version, project_name, index):
    best_json_name = os.path.join("bestmodels_" + project_name, "best.json")
    # from best.json import config
    with open(best_json_name, "r") as json_file:
        best_model_info = json.load(json_file)
    if best_model_info["finetune_model_layer"] == "mlp4":
        ft_model = mlp.MLP4()
    elif best_model_info["finetune_model_layer"] == "mlp6":
        ft_model = mlp.MLP6()
    else:
        raise ValueError("Unknown model configuration specified in best.json")    
    
    test_data_loader = get_data_loader(mode='test', batch_size=best_model_info["batch_size"], index=index)

    best_weight_name = os.path.join("bestweights_" + project_name, f"{model_version}.pkl")
    ft_model.set_state_dict(pdl.load(best_weight_name))   # 导入训练好的的模型权重
    ft_model.eval()
    all_result = []
    for (atom_bond_graph, bond_angle_graph, label_true_batch) in test_data_loader:
        label_predict_batch = ft_model(atom_bond_graph, bond_angle_graph)
        label_predict_batch = F.softmax(label_predict_batch)
        result = label_predict_batch[:, 1].cpu().numpy().reshape(-1).tolist()
        all_result.extend(result)
    nolabel_file_path = f'//8tb-disk/05.ZINC20_druglike/{index}_ZINC20_nolabel.csv'
    df = pd.read_csv(nolabel_file_path)
    # df = pd.read_csv('datasets/ZINC20_processed/test_nolabel.csv')
    df['pred'] = all_result
    result_file_path = 'datasets/DL_pred/result_' + project_name + '.csv'
    # 检查文件是否存在
    if index == 1:
        if os.path.exists(result_file_path):
            # 如果文件存在，则覆盖
            df.to_csv(result_file_path, index=False)
        else:
            # 如果文件不存在，则创建文件并写入数据
            df.to_csv(result_file_path, index=False)
    else: 
        if os.path.exists(result_file_path):
            # 如果文件存在，则追加数据
            df.to_csv(result_file_path, mode='a', header=False, index=False)
        else:
            # 如果文件不存在，则创建文件并写入数据
            df.to_csv(result_file_path, index=False)
    print(f'Screen through {index}_ZINC20_nolabel_' + project_name + '.csv')

    

# 将测试集的预测结果保存为result.csv
def test_DUDE(model_version, project_name, index):
    best_json_name = os.path.join("bestmodels_" + project_name, "best.json")
    # from best.json import config
    with open(best_json_name, "r") as json_file:
        best_model_info = json.load(json_file)
    if best_model_info["finetune_model_layer"] == "mlp4":
        ft_model = mlp.MLP4()
    elif best_model_info["finetune_model_layer"] == "mlp6":
        ft_model = mlp.MLP6()
    else:
        raise ValueError("Unknown model configuration specified in best.json")    
    
    test_data_loader = get_data_loader(mode='test', batch_size=best_model_info["batch_size"], index=index)

    best_weight_name = os.path.join("bestweights_" + project_name, f"{model_version}.pkl")
    ft_model.set_state_dict(pdl.load(best_weight_name))   # 导入训练好的的模型权重
    ft_model.eval()
    all_result = []
    for (atom_bond_graph, bond_angle_graph, label_true_batch) in test_data_loader:
        label_predict_batch = ft_model(atom_bond_graph, bond_angle_graph)
        label_predict_batch = F.softmax(label_predict_batch)
        result = label_predict_batch[:, 1].cpu().numpy().reshape(-1).tolist()
        all_result.extend(result)
    nolabel_file_path = '/DUD-E/'+project_name+ '/test_nolabel.csv'
    df = pd.read_csv(nolabel_file_path)
    df['pred'] = all_result
    result_file_path = 'datasets/DL_pred/result_' + project_name + '.csv'
    
    print(f'Screen through ' + project_name + '/test_nolabel.csv')
    