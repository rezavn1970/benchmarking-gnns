import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer
from layers.mlp_readout_layer import MLPReadout
from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class GATGINNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim_gat = net_params['hidden_dim_gat']
        hidden_dim_gin = net_params['hidden_dim_gin']

        out_dim_gat = net_params['out_dim_gat']
        out_dim_gin = net_params['out_dim_gin']

        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout



        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        self.embedding_h_gin = nn.Linear(in_dim, hidden_dim_gin)

        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim_gin, hidden_dim_gin, hidden_dim_gin)

            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, self.batch_norm, self.residual, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers + 1):
            self.linears_prediction.append(nn.Linear(hidden_dim_gin, n_classes))

        if self.readout == 'sum':
            self.pool = SumPooling()
        elif self.readout == 'mean':
            self.pool = AvgPooling()
        elif self.readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError



        self.embedding_h_gat = nn.Linear(in_dim, hidden_dim_gat * num_heads)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GATLayer(hidden_dim_gat * num_heads, hidden_dim_gat, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(GATLayer(hidden_dim_gat * num_heads, out_dim_gat, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim_gat, out_dim_gat)
        self.classifier = nn.Linear(83, n_classes)

    def forward(self, g, h, e, img):
        h_gat = self.embedding_h_gat(h)
        h_gat = self.in_feat_dropout(h_gat)
        for conv in self.layers:
            h_gat = conv(g, h_gat)
        g.ndata['h'] = h_gat

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes


#####################
        h_gin = self.embedding_h_gin(h)

        # list of hidden representation at each layer (including input)
        hidden_rep = [h_gin]

        for i in range(self.n_layers):
            h_gin = self.ginlayers[i](g, h_gin)
            hidden_rep.append(h_gin)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h_gin in enumerate(hidden_rep):
            pooled_h = self.pool(g, h_gin)
            # score_over_layer.append(self.linears_prediction[i](pooled_h))
            score_over_layer += self.linears_prediction[i](pooled_h)

        # print("reza is here")
        # print(type(score_over_layer))
        # print(self.MLP_layer(hg).shape)

        x = torch.cat((self.MLP_layer(hg), score_over_layer), dim=1)
        x = self.classifier(F.relu(x))




        return x

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss