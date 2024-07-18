# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import GAE
from torch_geometric.utils import dense_to_sparse
from MVPoollayers import GCN, MVPool
import random
from torch_geometric.data import Data
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from torch_geometric.nn import global_sort_pool
from torch_geometric.nn import SAGEConv
from pairwisesortpool import pairwisesortpool
from aug import remove_edge, drop_node

class ConvPool(torch.nn.Module):
    def __init__(self, args):
        super(ConvPool, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)
        
        self.conv4 = GCNConv(self.num_features, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.nhid)
        self.conv6 = GCNConv(self.nhid, self.nhid)
        
        self.conv7 = GCNConv(self.num_features, self.nhid)
        self.conv8 = GCNConv(self.nhid, self.nhid)
        self.conv9 = GCNConv(self.nhid, self.nhid)
        
        self.pool1 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool2 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool3 = MVPool(self.nhid, self.pooling_ratio, args)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        #xs = []
        #x_ = F.relu(self.conv4(x, edge_index, edge_attr))
        #xs.append(x_)
        #x_ = F.relu(self.conv5(x_, edge_index, edge_attr))
        #xs.append(x_)
        #x_ = F.relu(self.conv6(x_, edge_index, edge_attr))
        #xs.append(x_)
        #xs = torch.cat(xs, dim=1)

        x_aug_ = []
        for i in range(10):#找10个subgraph
            x_aug = drop_node(x, 0.1)
            x_aug = F.relu(self.conv7(x_aug, edge_index))
            x_aug = F.relu(self.conv8(x_aug, edge_index))
            x_aug = torch.cat([gmp(x_aug, batch), gap(x_aug, batch)], dim=1)
            if i != 0:
                x_aug_ = torch.cat([x_aug_, x_aug], dim=1)
            else:
                x_aug_ = x_aug
                
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool3(x, edge_index, edge_attr, batch)
                
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        return x, x_aug_


class ClassifierO(torch.nn.Module):
     def __init__(self, args):
        super(ClassifierO, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        
        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        
     def forward(self, x, y):  
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin3(x)
        x_ = F.log_softmax(x, dim=-1)
            
        loss = F.nll_loss(x_, y.long())
        pred = x_.max(dim=1)[1]
        
        return loss, pred, F.softmax(x,dim=1)

class ClassifierCross(torch.nn.Module):
     def __init__(self, args):
        super(ClassifierCross, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        
        self.lin1 = torch.nn.Linear(self.nhid * 4, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        
     def forward(self, x, y):  
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin3(x)
        x_ = F.log_softmax(x, dim=-1)
            
        loss = F.nll_loss(x_, y.long())
        pred = x_.max(dim=1)[1]
        
        return loss, pred, F.softmax(x,dim=1)  
       
    
class MIL(torch.nn.Module):
    def __init__(self, args):
        super(MIL, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.lossc = torch.nn.BCELoss()
        self.lin1 = torch.nn.Linear(self.nhid *2, self.nhid//2)
        self.lin2 = torch.nn.Linear(self.nhid//2, self.nhid // 8)
        self.lin3 = torch.nn.Linear(self.nhid // 8, 1)
        self.margin = 100
        
    def forward(self, x, y, training=True):  
        x = x.reshape(-1, self.nhid * 2)        
        x_norm = torch.norm(x, p=2, dim=1)
        x_norm = x_norm.reshape(-1, 20)
        x_select, x_slt_idx = torch.topk(x_norm, k=3, dim=1)
        x_select_avg = x_select.mean(dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = torch.sigmoid(self.lin3(x))
        x = x.reshape(-1, 20)
        
        score = torch.gather(x, 1, x_slt_idx)
        score = score.mean(dim=1)
        score = score.reshape(-1,1)   
        score = torch.clamp(score, min=1e-5, max=1.-1e-5)
        
        y = y.float()
        y = y.reshape(-1,1)
        
        if training==True:        
            loss_f_maj = x_select_avg[:y.shape[0]//2]
            loss_f_min = torch.abs(self.margin - x_select_avg[y.shape[0]//2:])
            loss_f = torch.mean( loss_f_maj + loss_f_min)
            loss = self.lossc(score, y) + 0.0001*loss_f
        else:
            loss = self.lossc(score, y)
        
        return loss, score
        
        
class ClassificationModel(torch.nn.Module):
    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        
        self.ConvPool = ConvPool(self.args)
        self.ClassifierO = ClassifierO(self.args)
        self.ClassifierCross = ClassifierCross(self.args)
        self.MIL = MIL(self.args)
        
        self.conv1 = SAGEConv(self.num_features, self.args.nhid)
        self.conv2 = SAGEConv(self.args.nhid, self.args.nhid)
        self.conv3 = SAGEConv(self.args.nhid, self.args.nhid)
        
        self.fc = torch.nn.Linear(self.nhid * 6, self.nhid * 3)
        #for m in self.modules():
        #    if isinstance(m, GCNConv):
        #        m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
         #       if m.bias is not None:
        #            m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, data, training=True):
        x, edge_index, batch, label = data.x, data.edge_index, data.batch, data.y        
        graph_rep, subg_rep = self.ConvPool(data)
        
        if training:
            index_min = []
            index_maj = []
            for i, y in enumerate(label):
                if y == 0:
                    index_maj.append(i)
                else:
                    index_min.append(i)
            graph_rep_maj = graph_rep[index_maj]            
            graph_rep_min = graph_rep[index_min]
            index_min_len = graph_rep_min.shape[0]
                        
            cross_num = min(index_min_len, len(index_maj))
            graph_rep_maj_ = graph_rep_maj[:cross_num,:]
            graph_rep_min_ = graph_rep_min[:cross_num,:]
            
            subg_rep_maj = subg_rep[index_maj]
            subg_rep_maj = subg_rep_maj[:cross_num,:]
            subg_rep_min = subg_rep[index_min]
            subg_rep_min = subg_rep_min[:cross_num,:]
            subg_rep_maj_ = torch.cat((subg_rep_maj,subg_rep_maj),dim=1)
            subg_rep_min_ = torch.cat((subg_rep_maj,subg_rep_min),dim=1)
            subg_rep_ = torch.cat((subg_rep_maj_,subg_rep_min_),dim=0)
            sub_label = torch.cat((torch.zeros(cross_num,device=self.args.device), torch.ones(cross_num,device=self.args.device)),dim=0)
            loss_sub, pred_sub = self.MIL(subg_rep_, sub_label, training=True)

            graph_rep_cross_maj = torch.cat((graph_rep_maj_, graph_rep_maj_), dim=1)

            cross_num_index = list(range(cross_num))
            random.shuffle(cross_num_index)
            cross_num_index = torch.tensor(cross_num_index, device=self.args.device)
            graph_rep_cross_maj_temp = torch.cat((graph_rep_maj_, graph_rep_maj_[cross_num_index]), dim=1)
            graph_rep_cross_maj = torch.cat((graph_rep_cross_maj, graph_rep_cross_maj_temp), dim=0)
            graph_rep_cross_maj_temp = torch.cat((graph_rep_maj_[cross_num_index], graph_rep_maj_), dim=1)
            graph_rep_cross_maj = torch.cat((graph_rep_cross_maj, graph_rep_cross_maj_temp), dim=0)
            
            graph_rep_cross_min = torch.cat((graph_rep_min_, graph_rep_min_), dim=1)
            graph_rep_cross_min_temp =  torch.cat((graph_rep_min_, graph_rep_maj_), dim=1)   
            graph_rep_cross_min = torch.cat((graph_rep_cross_min, graph_rep_cross_min_temp), dim=0)  
            graph_rep_cross_min_temp =  torch.cat((graph_rep_maj_, graph_rep_min_), dim=1)   
            graph_rep_cross_min = torch.cat((graph_rep_cross_min, graph_rep_cross_min_temp), dim=0)  
            
                
            
            graph_rep_total = torch.cat((graph_rep_maj, graph_rep_min), dim=0)
            label_total = torch.cat((torch.zeros((len(index_maj),), device=self.args.device), torch.ones((len(index_min),), device=self.args.device)), dim=0)
            loss_O, pred_O, logits_O = self.ClassifierO(graph_rep_total, label_total)



            
            graph_rep_cross_total = torch.cat((graph_rep_cross_maj, graph_rep_cross_min), dim=0)
            label_cross_total = torch.cat((torch.zeros((3*cross_num,), device=self.args.device),torch.ones((3*cross_num,), device=self.args.device)), dim=0)
            
            loss_cross, pred_cross, logits_cross = self.ClassifierCross(graph_rep_cross_total, label_cross_total)            
                        

            return loss_O + loss_cross + loss_sub
       
        else:
            loss_O, pred_O, logits_O = self.ClassifierO(graph_rep, data.y)
            
            rep_cross = torch.cat((graph_rep, graph_rep), dim=1)
            subg_cross = torch.cat((subg_rep, subg_rep), dim=1)
            loss_cross, pred_cross, logits_cross= self.ClassifierCross(rep_cross, data.y)
            loss_sub, pred_sub = self.MIL(subg_cross, data.y, training=False)
            
            logits_subg = torch.cat(((1- pred_sub.reshape(-1,1))*torch.ones([data.y.shape[0],1], device=self.args.device),pred_sub.reshape(-1,1)*torch.ones([data.y.shape[0],1], device=self.args.device)), dim=1)
            
            pred_final = (logits_O + logits_cross + logits_subg)/3
            pred_label = pred_final.max(dim=1)[1]
            correct = pred_label.eq(data.y).sum()
            return loss_O + loss_cross + loss_sub, pred_label, correct, pred_final       
 
