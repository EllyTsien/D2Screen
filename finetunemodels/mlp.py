import paddle as pdl
import paddle.nn as nn
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.networks.basic_block import MLP
import json

class MLP4(nn.Layer):
    def __init__(self):
        super(MLP4, self).__init__()
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


class MLP6(nn.Layer):
    def __init__(self):
        super(MLP6, self).__init__()
        compound_encoder_config = json.load(open('GEM/model_configs/geognn_l8.json', 'r'))  
        self.encoder = GeoGNNModel(compound_encoder_config) 
        self.encoder.set_state_dict(pdl.load("GEM/weight/class.pdparams")) 
        # GEM编码器输出的图特征为32维向量, 因此mlp的输入维度为32
        self.mlp = nn.Sequential(       
            nn.Linear(32, 64, weight_attr=nn.initializer.KaimingNormal()),  
            nn.ReLU(),
            nn.Linear(64, 128, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(128, 64, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(64, 32, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(32, 16, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(16, 2, weight_attr=nn.initializer.KaimingNormal()),
        )

    def forward(self, atom_bond_graph, bond_angle_graph):
        node_repr, edge_repr, graph_repr = self.encoder(atom_bond_graph.tensor(), bond_angle_graph.tensor())
        return self.mlp(graph_repr)
    

class DownstreamModel(nn.Layer):
    """
    Docstring for DownstreamModel,it is an supervised 
    GNN model which predicts the tasks shown in num_tasks and so on.
    """
    def __init__(self, model_config, compound_encoder):
        super(DownstreamModel, self).__init__()
        self.task_type = model_config['task_type']  # 或其他方法来初始化 task_type
        self.compound_encoder = compound_encoder
        self.compound_encoder.set_state_dict(pdl.load("GEM/weight/class.pdparams")) 
        self.norm = nn.LayerNorm(compound_encoder.graph_dim)
        self.mlp = MLP(
                model_config['layer_num'],
                in_size=compound_encoder.graph_dim,
                hidden_size=model_config['hidden_size'],
                out_size= 1,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate'])
        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()

    def forward(self, atom_bond_graphs, bond_angle_graphs):
        """
        Define the forward function,set the parameter layer options.compound_encoder 
        creates a graph data holders that attributes and features in the graph.
        Returns:
            pred: the model prediction.
        """
        node_repr, edge_repr, graph_repr = self.compound_encoder(atom_bond_graphs, bond_angle_graphs)
        graph_repr = self.norm(graph_repr)
        pred = self.mlp(graph_repr)
        if self.task_type == 'class':
            pred = self.out_act(pred)
        return pred


