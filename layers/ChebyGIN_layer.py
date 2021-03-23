import numpy as np
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers.attention_pooling_layer  import *
from layers.utils import *


class ChebyGINLayer(nn.Module):
    '''
    General Graph Neural Network layer that depending on arguments can be:
    1. Graph Convolution Layer (T. Kipf and M. Welling, ICLR 2017)
    2. Chebyshev Graph Convolution Layer (M. Defferrard et al., NeurIPS 2017)
    3. GIN Layer (K. Xu et al., ICLR 2019)
    4. ChebyGIN Layer (B. Knyazev et al., ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds)
    The first three types (1-3) of layers are particular cases of the fourth (4) case.
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 K,
                 n_hidden=0,
                 aggregation='mean',
                 activation=nn.ReLU(True),
                 n_relations=1):
        super(ChebyGINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_relations = n_relations
        assert K > 0, 'order is assumed to be > 0'
        self.K = K
        assert n_hidden >= 0, ('invalid n_hidden value', n_hidden)
        self.n_hidden = n_hidden
        assert aggregation in ['mean', 'sum'], ('invalid aggregation', aggregation)
        self.aggregation = aggregation
        self.activation = activation
        n_in = self.in_features * self.K * n_relations
        if self.n_hidden == 0:
            fc = [nn.Linear(n_in, self.out_features)]
        else:
            fc = [nn.Linear(n_in, n_hidden),
                  nn.ReLU(True),
                  nn.Linear(n_hidden, self.out_features)]
        if activation is not None:
            fc.append(activation)
        self.fc = nn.Sequential(*fc)
        print('ChebyGINLayer', list(self.fc.children())[0].weight.shape,
              torch.norm(list(self.fc.children())[0].weight, dim=1)[:10])

    def __repr__(self):
        return 'ChebyGINLayer(in_features={}, out_features={}, K={}, n_hidden={}, aggregation={})\nfc={}'.format(
            self.in_features,
            self.out_features,
            self.K,
            self.n_hidden,
            self.aggregation,
            str(self.fc))

    def chebyshev_basis(self, L, X, K):
        '''
        Return T_k X where T_k are the Chebyshev polynomials of order up to K.
        :param L: graph Laplacian, batch (B), nodes (N), nodes (N)
        :param X: input of size batch (B), nodes (N), features (F)
        :param K: Chebyshev polynomial order, i.e. filter size (number of hopes)
        :return: Tensor of size (B,N,K,F) as a result of multiplying T_k(L) by X for each order
        '''
        if K > 1:
            Xt = [X]
            Xt.append(torch.bmm(L, X))  # B,N,F
            for k in range(2, K):
                Xt.append(2 * torch.bmm(L, Xt[k - 1]) - Xt[k - 2])  # B,N,F
            Xt = torch.stack(Xt, 2)  # B,N,K,F
            return Xt
        else:
            # GCN
            assert K == 1, K
            return torch.bmm(L, X).unsqueeze(2)  # B,N,1,F

    def laplacian_batch(self, A, add_identity=False):
        '''
        Computes normalized Laplacian transformed so that its eigenvalues are in range [-1, 1].
        Note that sum of all eigenvalues = trace(L) = 0.
        :param A: Tensor of size (B,N,N) containing batch (B) of adjacency matrices of shape N,N
        :return: Normalized Laplacian of size (B,N,N)
        '''
        B, N = A.shape[:2]
        if add_identity:
            A = A + torch.eye(N, device=A.get_device() if A.is_cuda else 'cpu').unsqueeze(0)
        D = torch.sum(A, 1)  # nodes degree (B,N)
        D_hat = (D + 1e-5) ** (-0.5)
        L = D_hat.view(B, N, 1) * A * D_hat.view(B, 1, N)  # B,N,N
        if not add_identity:
            L = -L  # for ChebyNet to make a valid Chebyshev basis
        return D, L

    def forward(self, data):
        x, A, mask = data[:3]
        B, N, F = x.shape
        assert N == A.shape[1] == A.shape[2], ('invalid shape', N, x.shape, A.shape)

        if len(A.shape) == 3:
            A = A.unsqueeze(3)

        y_out = []
        for rel in range(A.shape[3]):
            D, L = self.laplacian_batch(A[:, :, :, rel], add_identity=self.K == 1)  # for the first layer this can be done at the preprocessing stage
            y = self.chebyshev_basis(L, x, self.K)  # B,N,K,F

            if self.aggregation == 'sum':
                # Sum features of neighbors
                if self.K == 1:
                    # GIN
                    y = y * D.view(B, N, 1, 1)
                else:
                    # ChebyGIN
                    D_GIN = torch.ones(B, N, self.K, device=x.get_device() if x.is_cuda else 'cpu')
                    D_GIN[:, :, 1:] = D.view(B, N, 1).expand(-1, -1, self.K - 1)  # keep self-loop features the same
                    y = y * D_GIN.view(B, N, self.K, 1)  # apply summation for other scales

            y_out.append(y)

        y = torch.cat(y_out, dim=2)
        y = self.fc(y.view(B, N, -1))  # B,N,F

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        y = y * mask.float()
        output = [y, A, mask]
        output.extend(data[3:] + [x])  # for python2

        return output