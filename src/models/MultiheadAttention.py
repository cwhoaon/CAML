import math
import sys
import torch

import numpy as np
import torch.nn as nn 
from collections import OrderedDict
from functools import partial
from torchvision.models.vision_transformer import MLPBlock
from typing import Any, Callable

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.types import _dtype as DType



from pyprojroot import here as project_root
sys.path.insert(0, str(project_root()))
import src.models.CAML as CAML

from src.datasets.dataloaders import FineTuneDataset
from torch.utils.data import DataLoader

class BitfitBias(nn.Module):
    def __init__(self, dim, n_bias=1, **factory_kwargs):
        super().__init__()
        self.dim = dim
        self.n_bias = n_bias
        self.q_bias = nn.Parameter(torch.zeros(n_bias, dim//3, **factory_kwargs))
        self.k_bias = nn.Parameter(torch.zeros(n_bias, dim//3, **factory_kwargs))
        self.v_bias = nn.Parameter(torch.zeros(n_bias, dim//3, **factory_kwargs))

    def forward(self, x, bias_idx):
        bsz, seq_len, _ = x.shape
        bias = torch.cat([self.q_bias[bias_idx], self.k_bias[bias_idx], self.v_bias[bias_idx]], dim=1).unsqueeze(1)
        return x + bias

class BitfitMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_bias,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype
        )
        self.num_bias = num_bias
        
        factory_kwargs = {"device": device, "dtype": dtype}
        if bias:
            self.in_proj_bitfit_bias = BitfitBias(3*embed_dim, n_bias=num_bias)
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bias_idx: Tensor,
        need_weights=False
    ): 
        bsz, seq_len, _ = query.shape
        
        proj = (query @ self.in_proj_weight.T).reshape(bsz, seq_len, -1)
        proj = self.in_proj_bitfit_bias(proj, bias_idx).reshape(bsz, seq_len, 3, -1)
        
        query = proj[:,:,0]
        key = proj[:,:,1]
        value = proj[:,:,2]
        
        def split_heads(tensor):
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, None


def split_data(x: list, y: list, max_samples, n_seeds, transform=None):
    x_s, y_s, x_q, y_q = [], [], [], []
    for _ in range(n_seeds):
        x_s_, y_s_, x_q_, y_q_ = [], [], [], []
        for c in range(len(x)):
            ids = torch.randperm(len(x[c][0]))
            k = len(ids) // 2
            ids_s = ids[:min(k, max_samples)]
            ids_q = ids[k:k + min(k, max_samples)]
            x_s_.append(x[c][0][ids_s])
            y_s_.append(y[c][0][ids_s])
            x_q_.append(x[c][0][ids_q])
            y_q_.append(y[c][0][ids_q])

        if transform is not None:
            x_s.append(transform(torch.cat(x_s_)))
            y_s.append(torch.cat(y_s_))
            x_q.append(transform(torch.cat(x_q_)))
            y_q.append(torch.cat(y_q_))
        else:    
            x_s.append(torch.cat(x_s_))
            y_s.append(torch.cat(y_s_))
            x_q.append(torch.cat(x_q_))
            y_q.append(torch.cat(y_q_))

    x_s = torch.stack(x_s)
    y_s = torch.stack(y_s)
    x_q = torch.stack(x_q)
    y_q = torch.stack(y_q)
    
    return x_s, y_s, x_q, y_q


if __name__=="__main__":
    embed_dim = 1536
    batch_size = 5
    num_heads = 16
    max_samples = 10
    num_bias = 5
    
    device = torch.device('cuda')
    dtype = torch.bfloat16
    
    domain = "wikiart_artist"
    split = 'train'
    task_path = f"/common_datasets/METAFLOW_DATASETS/task_descriptions/{domain}_embedding_{split}_500.pth"
    

    model = CAML.CAML(
        feature_extractor=nn.Identity(),
        fe_dim=1280,
        fe_dtype=dtype,
        train_fe=False, # whether to update the feature encoder weights during meta-training
        encoder_size="laion",
        num_bias=num_bias,
        dropout=0.0,
        label_elmes=True,
        device=device,
        set_transformer=False
    )
    model = model.to(device, dtype)

    
    dataset = FineTuneDataset(domain, split, task_path, None)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, pin_memory=True)   
    
    for i, batch in enumerate(dataloader):
        x, y = batch
        x_s, y_s, x_q, y_q = split_data(x, y, max_samples, num_bias)
        x_s, y_s, x_q, y_q = x_s.to(device=device, dtype=dtype), y_s.to(device=device), x_q.to(device=device, dtype=dtype), y_q.to(device=device)
        
        # x_s = x_s.unsqueeze(0)
        # y_s = y_s.unsqueeze(0)
        # x_q = x_q.unsqueeze(0)
        # y_q = y_q.unsqueeze(0)
        # inp = torch.cat([x_s, x_q], dim=0)
        # inp.to(device)
        # n_s = y_s.size(0)
        # output = model(inp)
        
        inp = torch.cat([x_s, x_q], dim=1)
        inp.to(device)
        n_s = y_s.size(1)
        bias_idx = torch.arange(num_bias).to(device)
        output = model(inp, y_s, n_s, bias_idx)
        
        print(output.shape)
        print(y_q.shape)
        
        
        break