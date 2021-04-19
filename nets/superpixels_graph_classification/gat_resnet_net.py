import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from torchvision import models

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer
from layers.mlp_readout_layer import MLPReadout

class GATResNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        count = 0
        self.resnet = models.resnet18(pretrained=True)
        
        for param in self.resnet.parameters():
            count = count + 1
            if count > 10:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
            
        self.resnet.fc = nn.Identity()

        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, out_dim)
        self.classifier = nn.Linear(out_dim+512, n_classes)

    def forward(self, g, h, e, img):
        
#         print("im inside forward")
#         print(type(img))
#         print(len(img))
#         print(type(g))
#         print(g.batch_size)        
        x2 = self.resnet( torch.stack(img))
    
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        x = torch.cat((self.MLP_layer(hg), x2), dim=1)
        x = self.classifier(F.relu(x))

        
        return x
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss