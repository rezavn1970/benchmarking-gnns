import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention_pooling_layer import *
from layers.utils import *
import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    Chebyshev Graph Convolution Layer (M. Defferrard et al., NeurIPS 2017)
    
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP

class ChebyGIN(nn.Module):
    '''
    Graph Neural Network class.
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 filters,
                 K=1,
                 n_hidden=0,
                 aggregation='mean',
                 dropout=0,
                 readout='max',
                 pool=None,  # Example: 'attn_gt_threshold_0_skip_skip'.split('_'),
                 pool_arch='fc_prev'.split('_'),
                 large_graph=False,  # > ~500 graphs
                 kl_weight=None,
                 graph_layer_fn=None,
                 init='normal',
                 scale=None,
                 debug=False):
        super(ChebyGIN, self).__init__()
        self.out_features = out_features
        assert len(filters) > 0, 'filters must be an iterable object with at least one element'
        assert K > 0, 'filter scale must be a positive integer'
        self.pool = pool
        self.pool_arch = pool_arch
        self.debug = debug
        n_prev = None

        attn_gnn = None
        if graph_layer_fn is None:
            graph_layer_fn = lambda n_in, n_out, K_, n_hidden_, activation: ChebyGINLayer(in_features=n_in,
                                                               out_features=n_out,
                                                               K=K_,
                                                               n_hidden=n_hidden_,
                                                               aggregation=aggregation,
                                                               activation=activation)
            if self.pool_arch is not None and self.pool_arch[0] == 'gnn':
                attn_gnn = lambda n_in: ChebyGIN(in_features=n_in,
                                                 out_features=0,
                                                 filters=[32, 32, 1],
                                                 K=np.min((K, 2)),
                                                 n_hidden=0,
                                                 graph_layer_fn=graph_layer_fn)

        graph_layers = []

        for layer, f in enumerate(filters + [None]):

            n_in = in_features if layer == 0 else filters[layer - 1]
            # Pooling layers
            # It's a non-standard way to put pooling before convolution, but it's important for our work
            if self.pool is not None and len(self.pool) > len(filters) + layer and self.pool[layer + 3] != 'skip':
                graph_layers.append(AttentionPooling(in_features=n_in, in_features_prev=n_prev,
                                                     pool_type=self.pool[:3] + [self.pool[layer + 3]],
                                                     pool_arch=self.pool_arch,
                                                     large_graph=large_graph,
                                                     kl_weight=kl_weight,
                                                     attn_gnn=attn_gnn,
                                                     init=init,
                                                     scale=scale,
                                                     debug=debug))

            if f is not None:
                # Graph "convolution" layers
                # no ReLU if the last layer and no fc layer after that
                graph_layers.append(graph_layer_fn(n_in, f, K, n_hidden,
                                                   None if self.out_features == 0 and layer == len(filters) - 1 else nn.ReLU(True)))
                n_prev = n_in

        if self.out_features > 0:
            # Global pooling over nodes
            graph_layers.append(GraphReadout(readout))
        self.graph_layers = nn.Sequential(*graph_layers)

        if self.out_features > 0:
            # Fully connected (classification/regression) layers
            self.fc = nn.Sequential(*(([nn.Dropout(p=dropout)] if dropout > 0 else []) + [nn.Linear(filters[-1], out_features)]))

    def forward(self, data):
        data = self.graph_layers(data)
        if self.out_features > 0:
            y = self.fc(data[0])  # B,out_features
        else:
            y = data[0]  # B,N,out_features
        return y, data[4]

class GraphReadout(nn.Module):
    '''
    Global pooling layer applied after the last graph layer.
    '''
    def __init__(self,
                 pool_type):
        super(GraphReadout, self).__init__()
        self.pool_type = pool_type
        dim = 1  # pooling over nodes
        if pool_type == 'max':
            self.readout_layer = lambda x, mask: torch.max(x, dim=dim)[0]
        elif pool_type in ['avg', 'mean']:
            # sum over all nodes, then divide by the number of valid nodes in each sample of the batch
            self.readout_layer = lambda x, mask: torch.sum(x, dim=dim) / torch.sum(mask, dim=dim).float()
        elif pool_type in ['sum']:
            self.readout_layer = lambda x, mask: torch.sum(x, dim=dim)
        else:
            raise NotImplementedError(pool_type)

    def __repr__(self):
        return 'GraphReadout({})'.format(self.pool_type)

    def forward(self, data):
        x, A, mask = data[:3]
        B, N = x.shape[:2]
        x = self.readout_layer(x, mask.view(B, N, 1))
        output = [x]
        output.extend(data[1:])   # [x, *data[1:]] doesn't work in Python2
        return output