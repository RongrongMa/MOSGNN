# -*- coding: utf-8 -*-

import torch 
from torch_geometric.utils import to_dense_batch

def pairwisesortpool(x, batch):
    fill_value = x.min().item() - 1
    
    batch_x, _ = to_dense_batch(x, batch, fill_value)
    B, N, D = batch_x.size()
    
    _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)

    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)
    
    batch_x_bo = batch_x[:, N-1:].contiguous()
    batch_x = batch_x[:, :1].contiguous()
    
    batch_x[batch_x == fill_value] = 0
    x = batch_x.view(B, D)
    
    
    batch_x_bo[batch_x_bo == fill_value] = 0
    x_bo = batch_x_bo.view(B, D)
    
    x = torch.cat((x, x_bo), dim=1)
    
    return x