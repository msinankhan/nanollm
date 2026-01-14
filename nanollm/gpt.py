from functools import partial
from dataclasses import dataclass 
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanollm.commons import get_dist_info, print0
from nanollm.muon import Muon, DistMuon
from nanollm.adamw import DistAdam


import os 
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from kernels import get_kernel
flash_attn=get_kernel('varunneal/flash-attention-3').flash_attn_interface

@dataclass
class GPTConfig:
    sequence_len: int = 1024 #Maximum number of tokens the model can see at once, defines the **context window**

    vocab_size: int = 50304

    n_layer: int = 12 # Number of Transformer blocks (depth) [Each layer has an attention block and an MLP block]

    n_head : int = 6 # Number of query heads

    n_kv_head : int = 6 # Number of key/value heads

    n_embd : int = 768 # Size of the residual stream, Everything lives in this space: embeddings, attention outputs, MLP outputs. This is the single most important dimension in the entire model.

    window_pattern : str = "L" # Controls sliding window attention per layer


def norm(x):
    #Normalization layers stabilize the signal flowing through the network. 
    #x ← x/√mean + epsilon. 
    return F.rms_norm(x,(x.size(-1),)) 


def apply_rotary_emd(x,cos,sin):
    assert x.ndim==4  # We expect the input tensor to be of the shape (B, T ,H, D), i.e Batch, Sequence_Length, Num_of_Heads, Head_Dimensions.
    d=x.shape[3]//2 
    x1,x2=x[...,:d],x[...,d:] # We split the head dimensions into pairs.
    y1= x1*cos + x2*sin             # It calculates as: y1[0]=x1[0]*cos+x2[0]* sin, y1[1]= x1[1]+x2[1]* sin and so on...
    y2=x1*(-sin) +x2*cos            #so we get the rotations between pairs as (x1[0],x2[0]), (x1[1],x2[1]) and so on.

    return torch.cat([y1,y2],3)