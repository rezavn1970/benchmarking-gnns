"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import numpy as np

from train.metrics import accuracy_MNIST_CIFAR as accuracy

from sklearn.metrics import confusion_matrix

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_images) in enumerate(data_loader):

        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        
        #batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        # for ganres:
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_images)        
        
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_images) in enumerate(data_loader):
            
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            
            #batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            # for ganres:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_images)

#             batch_score_max = batch_scores.detach().argmax(dim=1)
#             print ("batch score",batch_score_max)
#             print ("batch labels", batch_labels)
#             # Me: I am adding confusion matrix
#             results = confusion_matrix(batch_labels,batch_score_max)##(expected, predicted)
#             print("confusion matrix till here: ",results)
            
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc


#Me: I am adding this function to add at the end of the training
def evaluate_network_final(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    scores = []
    final_prediction =[] 
    flattened = []
    final_labels = []
    from itertools import chain
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            scores.append(batch_scores)
            
            
            # Me: adding final scores to return
            predict = batch_scores.detach().argmax(dim=1)
#             predict = predict.detach().cpu().numpy()[0]
            predict = predict.tolist()
            
            for item in predict:
                final_prediction.append(predict[item])
                
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
            
#             Me: prepare the labels to return 
            batch_labels = batch_labels.tolist()
            for sublabel in batch_labels:
                final_labels.append(sublabel)

            
#             print(predict)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc, final_prediction, scores, final_labels




"""
    For WL-GNNs
"""
def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)
        
        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels) 
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(scores, labels)
        nb_data += labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network_dense(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(scores, labels)
            nb_data += labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc