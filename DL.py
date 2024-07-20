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


import paddle.nn as nn
import models.mlp as mlp

#
from preprocess import Input_ligand_preprocess
from GEM import GEM_smile_transfer
from evaluation_train import evaluation_train

parser = ArgumentParser()
parser.add_argument('--model', default=None, type=str, help='Type of model to train (required)')
parser.add_argument('--dataset', default=None, type=str, help='Choose dataset (required)')
parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (default: 1e-3)')
parser.add_argument('--batch_size', default=32, type=int, help="Batch size (default: 32)")

def main(args):
    np.random.seed(args.seed)
    # torch.random.manual_seed(args.seed)

    if args.model is None:
        print("No model specified. Use '--model <model_name>'")
        exit(-1)

    if args.dataset is None:
        print("No dataset specified. Use '--dataset <dataset_name>'")
        exit(-1)
    else:
        input_ligands_path = 'datasets/' + args.dataset
        processed_input_path = 'datasets/train_preprocessed.csv'


    import pgl
    from pahelix.datasets.inmemory_dataset import InMemoryDataset
    import random
    from sklearn.model_selection import train_test_split

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


    def get_data_loader(mode, batch_size):
        if mode == 'train':
            # 训练模式下将train_data_list划分训练集和验证集，返回对应的DataLoader
            data_list = pkl.load(open(f'work/train_data_list.pkl', 'rb'))  # 读取data_list

            train_data_list, valid_data_list = train_test_split(data_list, test_size=0.2, random_state=42)
            print(f'train: {len(train_data_list)}, valid: {len(valid_data_list)}')

            train_dataset = InMemoryDataset(train_data_list)
            valid_dataset = InMemoryDataset(valid_data_list)
            train_data_loader = train_dataset.get_data_loader(
                batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_fn)
            valid_data_loader = valid_dataset.get_data_loader(
                batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_fn)
            return train_data_loader, valid_data_loader

        elif mode == 'test':
            # 推理模式下直接读取test_data_list, 返回test_data_loader
            data_list = pkl.load(open(f'work/test_data_list.pkl', 'rb'))

            test_dataset = InMemoryDataset(data_list)
            test_data_loader = test_dataset.get_data_loader(
                batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn)
            return test_data_loader
        
    def trial(model_version, model, batch_size, criterion, scheduler, opt):
        # 创建dataloader
        train_data_loader, valid_data_loader = get_data_loader(mode='train', batch_size=batch_size)   

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
        evaluator.plot_metrics([metric_train_list], [metric_valid_list])
        return train_metric_list, valid_metric_list

    processor = Input_ligand_preprocess(input_ligands_path)
    processor.preprocess() 
    processed_input_csv = pd.read_csv(processed_input_path)
    gem_smile_transfer = GEM_smile_transfer(processed_input_csv)
    gem_smile_transfer.run()
    
    
    # 固定随机种子
    SEED = 1024
    pdl.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)



    if args.model == "ADMET":
        model = mlp.ADMET() 
    else:
        raise NotImplementedError("Unknown model")


    batch_size = args.batch_size                                                             # batch size
    criterion = nn.CrossEntropyLoss()                                                   # 损失函数
    scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=15)         # 余弦退火学习率
    opt = optimizer.Adam(scheduler, parameters=model.parameters(), weight_decay=1e-5)   # 优化器

    metric_train_list, metric_valid_list = trial(model_version='1', model=model, batch_size=batch_size, criterion=criterion, scheduler=scheduler, opt=opt)
    
    # evaluator = evaluation_train(model, train_data_loader, valid_loader)
    # evaluator.plot_metrics([metric_train_list], [metric_valid_list])

'''
    if args.model == "simple_gnn":
        # Use node_coordinates (2dim) also as input feature
        model = SimpleGNN(2 + data.graph.nodes.shape[1] + trajectory_dim, gcn_dims=[16], fc_dims=[32, 256])
        path = '../best_model/' + args.model + '_' + args.dataset + '_' + str(args.min_history) + '_' + str(args.lr)  + '_' + str(args.batch_size) + '_' + args.trajectory_encoding + '_' + \
                str(model.gcn_dims) + '_'  + str(model.fc_dims) + '_best_model.pth'
    
    else:
        pass

'''

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
