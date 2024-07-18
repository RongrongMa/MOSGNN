# -*- coding: utf-8 -*-

import argparse
import glob
import os
import time
import torch
import torch.nn.functional as F
from models_graph import ClassificationModel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from sklearn.metrics import auc, precision_recall_curve, roc_curve, f1_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import random

parser = argparse.ArgumentParser()


parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--dataset', type=str, default=None, help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=200, help='patience for early stopping')
parser.add_argument('--regupara', type=float, default=0.5, help='regularization parameter')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=False, help='whether perform structure learning')
parser.add_argument('--hop_connection', type=bool, default=False, help='whether directly connect node within h-hops')
parser.add_argument('--hop', type=int, default=3, help='h-hops')
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--lamb', type=float, default=2.0, help='trade-off parameter')


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic=True
	

def train(model, train_loader, val_loader, args):
    max_prc = 1e-10
    patience_cnt = 0
    val_prc_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    current_epoch = 0
    for epoch in range(args.epochs):
        loss_train = 0.0
        current_epoch += 1
        for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(args.device)
                loss = model(data)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()            
        acc_val, loss_val, f1_val, auc_val, prc_val = compute_test(args, val_loader)
        if epoch%200 == 0:
            print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val),
              'auc_val: {:.6f}'.format(auc_val), 'prc_val: {:.6f}'.format(prc_val),'time: {:.6f}s'.format(time.time() - t))

        val_prc_values.append(prc_val)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_prc_values[-1] > max_prc:
            max_prc = val_prc_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(args, loader):
    label = []
    pred_label = []
    pred_logits = []
    model.eval()
    correct = 0.0
    loss_test = 0.0
    with torch.no_grad():        
        for j, data in enumerate(loader):
            data = data.to(args.device)
            loss, pred, cor, pred_logit = model(data, training=False)
            loss_test += loss.item()            
            correct += cor.item()
            
            if len(label) != 0:
                label = np.concatenate((label,data.y.cpu().detach().numpy()),axis=0)
                pred_label = np.concatenate((pred_label,pred.cpu().detach().numpy()),axis=0)
                pred_logits = np.concatenate((pred_logits,pred_logit.cpu().detach().numpy()),axis=0)
            else:
                label = data.y.cpu().detach().numpy()
                pred_label = pred.cpu().detach().numpy()
                pred_logits = pred_logit.cpu().detach().numpy()
             
        f1 = []        
        f1.append(f1_score(label, pred_label))
        f1.append(f1_score(label, pred_label, pos_label=0))
        f1.append(f1_score(label, pred_label, average='micro'))
        f1.append(f1_score(label, pred_label, average='macro'))
        
        
        fpr_ab, tpr_ab, _ = roc_curve(label, pred_logits[:,1])
        auroc = auc(fpr_ab, tpr_ab) 
        precision_ab, recall_ab, _ = precision_recall_curve(label, pred_logits[:,1])
        auprc = auc(recall_ab, precision_ab)   
        torch.cuda.empty_cache()
    return (correct/len(loader.dataset)), loss_test, f1, auroc, auprc

def compute_test_f1(args, loader, val_loader):
    label = []
    pred_logits = []
    label_val = []
    pred_logits_val = []
    model.eval()
    correct = 0.0
    loss_test = 0.0
    with torch.no_grad():        
        for j, data in enumerate(val_loader):
            data = data.to(args.device)
            _, _, _, pred_logit = model(data, training=False)          
            
            if len(label_val) != 0:
                label_val = np.concatenate((label_val,data.y.cpu().detach().numpy()),axis=0)
                pred_logits_val = torch.cat((pred_logits_val,pred_logit), dim=0)
            else:
                label_val = data.y.cpu().detach().numpy()
                pred_logits_val = pred_logit
        
        thred = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        label_min_val = torch.ones([pred_logits_val.shape[0]],device=args.device)
        label_maj_val = torch.zeros([pred_logits_val.shape[0]],device=args.device)
        f1_val = []
        for t in thred:
            pred_label_val = torch.where(pred_logits_val[:,1]>t, label_min_val, label_maj_val)
            f1_val.append(f1_score(label_val, pred_label_val.cpu().detach().numpy(), average='macro'))
        thred_ind = torch.tensor(f1_val).max(dim=-1)[1]    
        print(thred_ind)
        thred_best = thred[thred_ind]
        
        for j, data in enumerate(loader):
            data = data.to(args.device)
            loss, _, _, pred_logit = model(data, training=False)
            loss_test += loss.item()            
            
            if len(label) != 0:
                label = np.concatenate((label,data.y.cpu().detach().numpy()),axis=0)
                pred_logits = torch.cat((pred_logits,pred_logit),dim=0)
            else:
                label = data.y.cpu().detach().numpy()
                pred_logits = pred_logit
        label_min = torch.ones([pred_logits.shape[0]],device=args.device)
        label_maj = torch.zeros([pred_logits.shape[0]],device=args.device)
        pred_label = torch.where(pred_logits[:,1]>thred_best, label_min, label_maj)
        
        correct = pred_label.eq(torch.tensor(label,device=args.device)).sum().item()
        pred_label = pred_label.cpu().detach().numpy()
        f1 = []        
        f1.append(f1_score(label, pred_label))
        f1.append(f1_score(label, pred_label, pos_label=0))
        f1.append(f1_score(label, pred_label, average='micro'))
        f1.append(f1_score(label, pred_label, average='macro'))

        pred_logits = pred_logits.cpu().detach().numpy()
        fpr_ab, tpr_ab, _ = roc_curve(label, pred_logits[:,1])
        auroc = auc(fpr_ab, tpr_ab) 
        precision_ab, recall_ab, _ = precision_recall_curve(label, pred_logits[:,1])
        auprc = auc(recall_ab, precision_ab)   
        torch.cuda.empty_cache()
    return (correct/len(loader.dataset)), loss_test, f1, auroc, auprc




if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    for dataname in ['NCI1', 'NCI33','NCI41', 'NCI47','NCI81','NCI83', 'NCI109', 'NCI123', 'NCI145']:
            args.dataset = dataname
            dataset = TUDataset('data/', name=args.dataset, use_node_attr=True)
            args.num_classes = dataset.num_classes
            args.num_features = dataset.num_features

            graph_label = [int(data.y.cpu()) for data in dataset]
            kfd=StratifiedKFold(n_splits=3, random_state=args.seed, shuffle=True)#######n_splits  
    
            final_test_acc = []
            final_test_loss = []
            final_test_f1 = []
            final_test_auc = []
            final_test_prc = []
            for fold_number, (train_index,test_index) in enumerate(kfd.split(dataset, graph_label)):
                graphs_train = [dataset[int(i)] for i in train_index]
                graphs_test = [dataset[int(i)] for i in test_index]                        
                graphs_train_label = [int(data.y.cpu()) for data in graphs_train]
                graphs_test_label = [int(data.y.cpu()) for data in graphs_test] 
                print(Counter(graphs_train_label), Counter(graphs_test_label))
            
                training_set, validation_set, y_train, _ = train_test_split(graphs_train,graphs_train_label,test_size=0.2,random_state=42,stratify=graphs_train_label)

                graph_index = torch.tensor(range(len(training_set)))
                ros = RandomOverSampler(random_state=42)
                dataset_resampled, _ = ros.fit_resample(graph_index.reshape([len(training_set),1]), y_train)

                graph_dataset = []
                dataset_resampled = dataset_resampled.reshape([-1])
                for i in dataset_resampled:
                    graph_dataset.append(graph_train[int(i)])
                args.avg_num_nodes =  np.ceil(np.mean([data.num_nodes for data in graph_train]))
                        
                train_loader = DataLoader(graph_dataset, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
                test_loader = DataLoader(graphs_test, batch_size=args.batch_size, shuffle=False)

                model = ClassificationModel(args).to(args.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)        
                best_model = train(model, train_loader, val_loader, args)
                model.load_state_dict(torch.load('{}.pth'.format(best_model)))
                test_acc, test_loss, test_f1, test_auc, test_prc = compute_test_f1(args, test_loader, val_loader)
                print('Test set results, loss = {:.6f}, accuracy = {:.6f}, f1 score ={}, auc = {:.6f}, prc = {:.6f}'.format(test_loss, test_acc, test_f1, test_auc, test_prc))
                final_test_acc.append(test_acc)
                final_test_loss.append(test_loss)
                final_test_f1.append(test_f1)            
                final_test_auc.append(test_auc)
                final_test_prc.append(test_prc)

            final_test_acc = np.array(final_test_acc)
            final_test_loss = np.array(final_test_loss)
            final_test_f1 = np.array(final_test_f1)     
            final_test_auc = np.array(final_test_auc)
            final_test_prc = np.array(final_test_prc)
            acc_mean = final_test_acc.mean()
            acc_std = final_test_acc.std() 
            loss_mean = final_test_loss.mean()
            loss_std = final_test_loss.std() 
            f1_1_mean = final_test_f1[:,0].mean()
            f1_1_std = final_test_f1[:,0].std() 
            f1_0_mean = final_test_f1[:,1].mean()
            f1_0_std = final_test_f1[:,1].std() 
            f1_micro_mean = final_test_f1[:,2].mean()
            f1_micro_std = final_test_f1[:,2].std() 
            f1_macro_mean = final_test_f1[:,3].mean()
            f1_macro_std = final_test_f1[:,3].std() 
            auc_mean = final_test_auc.mean()
            auc_std = final_test_auc.std() 
            prc_mean = final_test_prc.mean()
            prc_std = final_test_prc.std() 
            print('Final test set results, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f} '.format(args.dataset,acc_mean,acc_std,loss_mean,loss_std, f1_1_mean, f1_1_std, f1_0_mean, f1_0_std, f1_micro_mean, f1_micro_std, f1_macro_mean,f1_macro_std,auc_mean,auc_std, prc_mean, prc_std))

