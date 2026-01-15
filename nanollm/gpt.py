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

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx=layer_idx #KV Cache is shared across layers, each layer has its own KV cache. 
        self.n_embed=config.n_embed
        self.n_head=config.n_head
        self.n_kv_head=config.n_kv_head
        self.head_dim=self.n_embed//self.n_head

        assert self.n_embed%self.n_head==0
        assert self.n_kv_head<=self.n_head and self.n_head% self.n_kv_head==0 #We can have fewer K/V heads than Q heads, but every K/V head must have an equal num of Q heads

        self.c_q=nn.Linear(self.n_embed, self.n_head*self.head_dim, bias=False)  #nn.Linear perform as Affine Transformation i.e y=x.W^T
        self.c_k=nn.Linear(self.n_embed,self.n_kv_head*self.head_dim, bias=False) # these 4 lines intiatialize a weight matrix W^T of the (out_features, in_features).
        self.c_v=nn.Linear(self.n_embed, self.n_kv_head*self.head_dim,bias=False) #When we want to project the matrices later, this matrix W^T is multiplied with the input.
        self.c_proj=nn.Linear(self.n_embed,self.n_embed, bias=False) #The weight matrix is initialized as a He initialization, drawing values from a uniform distribution U(√-k, √k), k=1/in_features


    def forward(self,x,cos_sin, window_size,kv_cache):
        B,T,C=x.size() #C is n_embed, the width of the model

        q=self.c_q(x).view(B, T, self.n_head, self.head_dim) #c_q(x) performs the linear projection of the matrix. using the weight matrix defined in c_q.
        k=self.c_k(x).view(B,T,self.n_kv_head,self.head_dim) # .view() changes the shape of the tensor. from (B,T,128) to (B,T,8,16), 
        v=self.c_v(x).view(B,T, self.n_kv_head,self.head_dim) # just adds newer rows, doesn't change the values. 

        cos,sin=cos_sin

        q,k=apply_rotary_emd(q,cos,sin), apply_rotary_emd(k,cos,sin) # Rotates the Q & K. 

        q,k=norm(q),norm(k)

        if kv_cache is None:
            #Training
            y=flash_attn.flash_attn_func(q,k,v,causal=True, window_size=window_size)

        else:
            #Inference
            k_cache,v_cache=kv_cache.get_layer_cache(self.layer_idx)
            y=flash_attn.flash_attn_with_kvcache(
                q,k_cache,v_cache
                k=k,v=v,
                cache_seqlens=kv_cache.seqlens,
                causal=True,
                window_size=window_size)
            
            if self.layer_idx==kv_cache.n_layers-1:
                kv_cache.advance(T) # Every layer edits the same slot in the kv_cache, once we are at the last layer, we gotta move the write pointer forward by Token T.
                                    # This is to prevent over writing the same token.


        y=y.contiguous().view(B,T,-1) # Reshape it back to the original tensor shape. flash Atten gives: (B, T, H, D), here we flatten it back to (B, T, n_embd), n_embed=H*D
        y=self.c_proj(y) # Remixes the head back into the residual stream.

        return y