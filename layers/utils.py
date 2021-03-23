import numpy as np
import os
import torch
import copy
# from graphdata import *
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
import numbers
import random


def load_save_noise(f, noise_shape):
    if os.path.isfile(f):
        print('loading noise from %s' % f)
        noises = torch.load(f)
    else:
        noises = torch.randn(noise_shape, dtype=torch.float)
        # np.save(f, noises.numpy())
        torch.save(noises, f)
    return noises


def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data


def data_to_device(data, device):
    if isinstance(data, dict):
        keys = list(data.keys())
    else:
        keys = range(len(data))
    for i in keys:
        if isinstance(data[i], list) or isinstance(data[i], dict):
            data[i] = data_to_device(data[i], device)
        else:
            if isinstance(data[i], torch.Tensor):
                try:
                    data[i] = data[i].to(device)
                except:
                    print('error', i, data[i], type(data[i]))
                    raise
    return data


def count_correct(output, target, N_nodes=None, N_nodes_min=0, N_nodes_max=25):
    if output.shape[1] == 1:
        # Regression
        pred = output.round().long()
    else:
        # Classification
        pred = output.max(1, keepdim=True)[1]
    target = target.long().squeeze().cpu()  # for older pytorch
    pred = pred.squeeze().cpu()  # for older pytorch
    if N_nodes is not None:
        idx = (N_nodes >= N_nodes_min) & (N_nodes <= N_nodes_max)
        if idx.sum() > 0:
            correct = pred[idx].eq(target[idx]).sum().item()
            for lbl in torch.unique(target, sorted=True):
                idx_lbl = target[idx] == lbl
                eq = (pred[idx][idx_lbl] == target[idx][idx_lbl]).float()
                print('lbl: {}, avg acc: {:2.2f}% ({}/{})'.format(lbl, 100 * eq.mean(), int(eq.sum()),
                                                                  int(idx_lbl.float().sum())))

            eq = (pred[idx] == target[idx]).float()
            print('{} <= N_nodes <= {} (min={}, max={}), avg acc: {:2.2f}% ({}/{})'.format(N_nodes_min,
                                                                                           N_nodes_max,
                                                                                           N_nodes[idx].min(),
                                                                                           N_nodes[idx].max(),
                                                                                           100 * eq.mean(),
                                                                                           int(eq.sum()),
                                                                                           int(idx.sum())))
        else:
            correct = 0
            print('no graphs with nodes >= {} and <= {}'.format(N_nodes_min, N_nodes_max))
    else:
        correct = pred.eq(target).sum().item()

    return correct


def attn_AUC(alpha_GT, alpha):
    auc = []
    if len(alpha) > 0 and alpha_GT is not None and len(alpha_GT) > 0:
        for layer in alpha:
            alpha_gt = np.concatenate([a.flatten() for a in alpha_GT[layer]]) > 0
            if len(np.unique(alpha_gt)) <= 1:
                print('Only one class ({}) present in y_true. ROC AUC score is not defined in that case.'.format(
                    np.unique(alpha_gt)))
                auc.append(np.nan)
            else:
                auc.append(100 * roc_auc_score(y_true=alpha_gt,
                                               y_score=np.concatenate([a.flatten() for a in alpha[layer]])))
    return auc


def stats(arr):
    return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)


def normalize(x, eps=1e-7):
    return x / (x.sum() + eps)


def normalize_batch(x, dim=1, eps=1e-7):
    return x / (x.sum(dim=dim, keepdim=True) + eps)


def normalize_zero_one(im, eps=1e-7):
    m1 = im.min()
    m2 = im.max()
    return (im - m1) / (m2 - m1 + eps)


def mse_loss(target, output, reduction='mean', reduce=None):
    loss = (target.float().squeeze() - output.float().squeeze()) ** 2
    if reduce is None:
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
        else:
            NotImplementedError(reduction)
    elif not reduce:
        return loss
    else:
        NotImplementedError('use reduction if reduce=True')


def shuffle_nodes(batch):
    x, A, mask, labels, params_dict = batch
    for b in range(x.shape[0]):
        idx = np.random.permutation(x.shape[1])
        x[b] = x[b, idx]
        A[b] = A[b, :, idx][idx, :]
        mask[b] = mask[b, idx]
        if 'node_attn' in params_dict:
            params_dict['node_attn'][b] = params_dict['node_attn'][b, idx]
    return [x, A, mask, labels, params_dict]


def copy_batch(data):
    data_cp = []
    for i in range(len(data)):
        if isinstance(data[i], dict):
            data_cp.append({key: data[i][key].clone() for key in data[i]})
        else:
            data_cp.append(data[i].clone())
    return data_cp


def sanity_check(model, data):
    with torch.no_grad():
        output1 = model(copy_batch(data))[0]
        output2 = model(shuffle_nodes(copy_batch(data)))[0]
        if not torch.allclose(output1, output2, rtol=1e-02, atol=1e-03):
            print('WARNING: model outputs different depending on the nodes order', (torch.norm(output1 - output2),
                                                                                    torch.max(output1 - output2),
                                                                                    torch.max(output1),
                                                                                    torch.max(output2)))
    print('model is checked for nodes shuffling')


def set_seed(seed, seed_data=None):
    random.seed(seed)  # for some libraries
    rnd = np.random.RandomState(seed)
    if seed_data is not None:
        rnd_data = np.random.RandomState(seed_data)
    else:
        rnd_data = rnd
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return rnd, rnd_data


def compute_feature_stats(model, train_loader, device, n_batches=100):
    print('computing mean and std of input features')
    model.eval()
    x = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            x.append(data[0].data.cpu().numpy())  # B,N,F
            if batch_idx > n_batches:
                break
    x = np.concatenate(x, axis=1).reshape(-1, x[0].shape[-1])
    print('features shape loaded', x.shape)

    mn = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    print('mn', mn)
    print('std', sd)
    sd[sd < 1e-2] = 1  # to prevent dividing by a small number
    print('corrected (non zeros) std', sd)  # .data.cpu().numpy())

    mn = torch.from_numpy(mn).float().to(device).unsqueeze(0)
    sd = torch.from_numpy(sd).float().to(device).unsqueeze(0)
    return mn, sd


def copy_data(data, idx):
    data_new = {}
    for key in data:
        if key == 'Max_degree':
            data_new[key] = data[key]
            print(key, data_new[key])
        else:
            data_new[key] = copy.deepcopy([data[key][i] for i in idx])
            if key in ['graph_labels', 'N_edges']:
                data_new[key] = np.array(data_new[key], np.int32)
            print(key, len(data_new[key]))

    return data_new


def concat_data(data):
    data_new = {}
    for key in data[0]:
        if key == 'Max_degree':
            data_new[key] = np.max(np.array([d[key] for d in data]))
            print(key, data_new[key])
        else:
            if key in ['graph_labels', 'N_edges']:
                data_new[key] = np.concatenate([d[key] for d in data])
            else:
                lst = []
                for d in data:
                    lst.extend(d[key])
                data_new[key] = lst
            print(key, len(data_new[key]))

    return data_new