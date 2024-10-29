import paddle as pdl
import paddle.nn as nn
from pahelix.model_zoo.gem_model import GeoGNNModel
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
            nn.Linear(128, 64, weight_attr=nn.initializer.KaimingNormal()),  
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
