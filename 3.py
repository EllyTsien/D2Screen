import pickle as pkl
import pandas as pd
import numpy as np
# 测试集
test_df = pd.read_csv('datasets/small_file_1.csv') #test dataset 路径 ########################change
smiles_list = test_df["SMILES"].tolist()
pkl.dump(smiles_list, open('work_3/test_smiles_list.pkl', 'wb')) ########################change

# 将smiles转化为rdkit标准smiles
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # 屏蔽RDKit的warning

for index, row in test_df.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['SMILES'])
        new_smiles = Chem.MolToSmiles(mol)
        test_df.loc[index, 'SMILES'] = new_smiles
    except:
        # 若转化失败，则认为原始smile不合法，删除该数据
        test_df.drop(index, inplace=True)

print(f'len of test_df is {len(test_df)}')

#test dataset convert
# 使用分子力场将smiles转化为3d分子图，并保存为smiles_to_graph_dict.pkl文件
from threading import Thread, Lock
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from rdkit.Chem import AllChem

import codecs
import csv
def data_write_remain_test_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
        file_csv = codecs.open(file_name,'w+','utf-8')#追加
        writer = csv.writer(file_csv)
        writer.writerow(['SMILES'])
        for data in datas:
            writer.writerow([data])
        print("保存csv文件成功，处理结束")

mutex = Lock()  # 互斥锁，防止多个线程同时修改某一文件或某一全局变量，引发未知错误

def calculate_3D_structure_(smiles_list):
    n = len(smiles_list)
    global p
    index = 0
    while True:
        mutex.acquire()  # 获取锁
        if p >= n:
            mutex.release()
            break
        index = p        # p指针指向的位置为当前线程要处理的smiles
        smiles = smiles_list[index]
        print(index, ':', round(index / n * 100, 2), '%', smiles)
        p += 1           # 修改全局变量p
        mutex.release()  # 释放锁
        try:
            molecule = AllChem.MolFromSmiles(smiles)
            molecule_graph = mol_to_geognn_graph_data_MMFF3d(molecule)  # 根据分子力场生成3d分子图
        except:
            print("Invalid smiles!", smiles)
            mutex.acquire()
            with open('work_3/invalid_smiles.txt', 'a') as f: ########################change
                # 生成失败的smiles写入txt文件保存在该目录下
                f.write(str(smiles) + '\n')
            mutex.release()
            continue

        global smiles_to_graph_dict
        mutex.acquire()   # 获取锁
        smiles_to_graph_dict[smiles] = molecule_graph
        mutex.release()   # 释放锁

for mode in ['test']:
    smiles_list = test_df["SMILES"].tolist()
    data_write_remain_test_csv('2_remain_test.csv', smiles_list)   ########################change
    global smiles_to_graph_dict
    smiles_to_graph_dict = {}
    global p              # p为全局指针，指向即将要处理的smiles
    p = 0
    thread_count = 12      # 线程数。一般根据当前运行环境下cpu的核数来设定
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_3D_structure_, args=(smiles_list, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_graph_dict, open(f'work_3/{mode}_smiles_to_graph_dict.pkl', 'wb'))
    print(f'{mode} is Done!')

# 将smiles、graph、label构建成一个列表，并保存为data_list.pkl文件，该文件为GEM读取的数据文件
# for test
# train_smiles_to_graph_dict = pkl.load(open(f'work/train_smiles_to_graph_dict.pkl','rb'))
test_smiles_to_graph_dict = pkl.load(open(f'work_3/test_smiles_to_graph_dict.pkl','rb'))

# train_data_list = []
test_data_list = []

'''
for index, row in train_df.iterrows():
    smiles = row["SMILES"]
    label = row["label"]
    if smiles not in train_smiles_to_graph_dict:
        continue
    data_item = {
        "smiles": smiles,
        "graph": train_smiles_to_graph_dict[smiles],
        "label": label,
    }
    train_data_list.append(data_item)
'''

for index, row in test_df.iterrows():
    smiles = row["SMILES"]
    if smiles not in test_smiles_to_graph_dict:
        continue
    data_item = {
        "smiles": smiles,
        "graph": test_smiles_to_graph_dict[smiles],
        'label': 0
    }
    test_data_list.append(data_item)

# pkl.dump(train_data_list, open('work/train_data_list.pkl', 'wb'))
pkl.dump(test_data_list, open('work_3/test_data_list.pkl', 'wb'))

######################################################################################################################################
# 定义模型
import pgl
from pahelix.datasets.inmemory_dataset import InMemoryDataset
import random
from sklearn.model_selection import train_test_split


import paddle as pdl
import paddle.nn as nn
from pahelix.model_zoo.gem_model import GeoGNNModel
import json

class ADMET(nn.Layer):
    def __init__(self):
        super(ADMET, self).__init__()
        compound_encoder_config = json.load(open('GEM/model_configs/geognn_l8.json', 'r'))  
        self.encoder = GeoGNNModel(compound_encoder_config) 
        self.encoder.set_state_dict(pdl.load("GEM/weight/class.pdparams")) 
        # GEM编码器输出的图特征为32维向量, 因此mlp的输入维度为32
        self.mlp = nn.Sequential(       
            nn.Linear(32, 32, weight_attr=nn.initializer.KaimingNormal()),  
            nn.ReLU(),
            nn.Linear(32, 32, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(32, 32, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(32, 2, weight_attr=nn.initializer.KaimingNormal()),
        )

    def forward(self, atom_bond_graph, bond_angle_graph):
        node_repr, edge_repr, graph_repr = self.encoder(atom_bond_graph.tensor(), bond_angle_graph.tensor())
        return self.mlp(graph_repr)

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
        data_list = pkl.load(open(f'work_3/train_data_list.pkl', 'rb'))  # 读取data_list

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
        data_list = pkl.load(open(f'work_3/test_data_list.pkl', 'rb'))

        test_dataset = InMemoryDataset(data_list)
        test_data_loader = test_dataset.get_data_loader(
            batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn)
        return test_data_loader
        
import pickle as pkl
import pandas as pd
import numpy as np

print('start prediction process')

# 将测试集的预测结果保存为result.csv
import paddle.nn.functional as F

import codecs
import csv

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
        file_csv = codecs.open(file_name,'w+','utf-8')#追加
        writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for data in datas:
            writer.writerow(str(data))
        print("保存csv文件成功，处理结束")


def test(model_version):
    test_data_loader = get_data_loader(mode='test', batch_size=1024)

    model = ADMET()
    model.set_state_dict(pdl.load("weight/" + model_version + ".pkl"))   # 导入训练好的的模型权重
    model.eval()

    all_result = []
    for (atom_bond_graph, bond_angle_graph, label_true_batch) in test_data_loader:
        label_predict_batch = model(atom_bond_graph, bond_angle_graph)
        label_predict_batch = F.softmax(label_predict_batch)
        result = label_predict_batch[:, 1].cpu().numpy().reshape(-1).tolist()
        all_result.extend(result)

    # print(all_result)
    data_write_csv('2_pred_result.csv',all_result)  ########################change

    # for i in test_data_loader:
    #     print(i)


    df2 = pd.read_csv('2_remain_test.csv')  ########################change
    df2[f'pred'] = all_result
    df2.to_csv('2_result.csv', index=False)     ########################change

test(model_version='1')


