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