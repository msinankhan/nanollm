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
    sequence_len: int = 2048 #Maximum number of tokens the model can see at once, defines the **context window**

    vocab_size: int = 32768

    n_layer: int = 12 # Number of Transformer blocks (depth) [Each layer has an attention block and an MLP block]

    n_head : int = 6 # Number of query heads

    n_kv_head : int = 6 # Number of key/value heads

    n_embd : int = 768 # Size of the residual stream, Everything lives in this space: embeddings, attention outputs, MLP outputs. This is the single most important dimension in the entire model.

    window_pattern : str = "SSSL" # Controls sliding window attention per layer


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

def has_ve(layer_idx,n_layer):
    return layer_idx%2==(n_layer-1)%2

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

        self.ve_gate_channels=32 # Can increase to 64 for deeper models.
        self.ve_gate=nn.Linear(self.ve_gate_channels, self.n_kv_head,bias=False) if has_ve(layer_idx,config.n_layer) else None


    def forward(self,x,ve, cos_sin, window_size,kv_cache):
        B,T,C=x.size() #C is n_embed, the width of the model

        q=self.c_q(x).view(B, T, self.n_head, self.head_dim) #c_q(x) performs the linear projection of the matrix. using the weight matrix defined in c_q.
        k=self.c_k(x).view(B,T,self.n_kv_head,self.head_dim) # .view() changes the shape of the tensor. from (B,T,128) to (B,T,8,16), 
        v=self.c_v(x).view(B,T, self.n_kv_head,self.head_dim) # just adds newer rows, doesn't change the values. 


        if ve is not None:
            ve=ve.view(B,T,self.n_kv_head,self.head_dim)
            gate=2* torch.sigmoid(self.ve_gate(x[...,:self.ve_gate_channels])) # We only take the first 32 elements of the vector x as input to the linear layer in ve_gate. 
            v=v+gate.unsqueeze(-1)*ve # .unsqueeze() Broadcasts (B,T,n_kv_head,head_dim) to (B,T,n_kv_head,1) so it scales the entire ve vector per head.

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
                q,k_cache,v_cache,
                k=k,v=v,
                cache_seqlens=kv_cache.seqlens,
                causal=True,
                window_size=window_size  )
            
            if self.layer_idx==kv_cache.n_layers-1:
                kv_cache.advance(T) # Every layer edits the same slot in the kv_cache, once we are at the last layer, we gotta move the write pointer forward by Token T.
                                    # This is to prevent over writing the same token.


        y=y.contiguous().view(B,T,-1) # Reshape it back to the original tensor shape. flash Atten gives: (B, T, H, D), here we flatten it back to (B, T, n_embd), n_embed=H*D
        y=self.c_proj(y) # Remixes the head back into the residual stream.

        return y
    

class MLP(nn.Module):
    def __init__(self, config ):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embed,4*config.n_embed,bias=False) # Increase the dimensions so that the model has an increased representational capability.
                                                                        # It provides sufficient capacity for the model to mix and transform features 
        self.c_proj=nn.Linear(4*config.n_embed,config.n_embed,bias=False) # Bias is often unnecessary when we use normalization and also reduces optimization issues.
        
    def forward(self,x):
        # (B, T, n_embd) → (B, T, 4 * n_embd)
        x=self.c_fc(x) 

        x=F.relu(x).square() #This creates a quadratic growth for positive values, making the activation smoother and more expressive than plain ReLU

       
        #This "down-projection" compresses the activated features back to the original dimension, 
        # summarizing the computations for the residual stream. 
        # It acts as a learned aggregation, **allowing the model to select which expanded features matter**.

        x=self.c_proj(x)  # (B, T, 4 * n_embd) → (B, T, n_embd)
        return x


class Block(nn.Module):
    def __init__(self, config,layer_idx):
        super().__init__()
        self.attn=CausalSelfAttention(config,layer_idx)
        self.mlp=MLP(config)

    def forward(self,x,ve,cos_sin,window_size,kv_cache):
        x=x+self.attn(norm(x),ve, cos_sin,window_size,kv_cache)
        x=x+self.mlp(norm(x))

        return x
    

class GPT(nn.Module):
    def __init__(self,config,pad_vocab_size_to=64):
        super().__init__()
        self.config=config
        self.window_sizes=self._compute_window_sizes(config) #Computes (left,right) number of tokens the attn can see. (-1,0) for full context.

        padded_vocab_size=((config.vocab_size + pad_vocab_size_to - 1 )//pad_vocab_size_to) * pad_vocab_size_to # rounds the vocab size to be a multiple of 64 for GPU efficiency.

        if padded_vocab_size!=config.vocab_size:
            print0(f"Padding vocab size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer=nn.ModuleDict({       #Module Dict is like a python Dictionary for ML models, we use it over python's dict as it tracks parameters. 
            "wte":nn.Embedding(padded_vocab_size,config.n_embed), #Randomly initializes a matrix of shape (pd_vocab_size,n_embd) with a Normal dist.
            "h":nn.ModuleList([Block(config,layer_idx) for layer_idx in range(config.n_layer)]) # Module List is like a python list
        })                                                                                      # But it also tracks parameters.


        self.lm_head=nn.Linear(config.n_embed, padded_vocab_size,bias=False) #This layer is to turn the hidden state into logits.
                                                                            # So we get back the vocab_size vectors


        #The following are per-layer, learnable scalar gates that **control how information flows** through depth.
        #At layer l, the update looks like: x(ℓ+1) ​ =λresid (ℓ)⋅ x (ℓ) ​ + λx0(ℓ)⋅x0 + Block(x ℓ)
        #x0= Original Embeddings
        #x(ℓ) = Current hidden state
        #Block= attn + MLP
        # FAKE INIT: META DEVICE CONTEXT.
        self.resid_lambda=nn.Parameter(torch.ones(config.n_layer)) # At initialization, we have x(ℓ+1​)=x(ℓ)​+f(x(ℓ​))
        self.x0_lambdas=nn.Parameter(torch.zeros(config.n_layer)) # This is needed because, without x0, the information from the earlier layer drifts.      
                                                                  # As in the classical transformer, we only add previous hidden layers i.e , x(ℓ+1)= x(ℓ) + f(x(ℓ))

        self.rotate_seq_len= config.sequence_len*10  # We allocate the cache for the rotary embeddings, to store it before hand so that we don't have to compute it in forward.()

        head_dim=config.n_embed//config.n_head #
        kv_dim=config.n_kv_head*head_dim
        self.value_embeds=nn.ModuleDict({str(i):nn.Embedding(padded_vocab_size,kv_dim) for i in range(config.n_layer) if has_ve(i,config.n_layer)}) #Res-transformer style mixing of V, to have better signal propagation. This dictionary holds value embeddings from alternating layers.

        cos,sin=self._precompute_rotary_embeddings(self.rotate_seq_len, head_dim)

        self.register_buffer("cos",cos,persistent=False)  # We store cos and sin in buffers because we don't want them to be trained upon as they are determined by the relative 
        self.register_buffer("sin",sin, persistent=False) # position of the tokens. Hence the gradient is not calculated over it.

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0) #As the embedding layer is simply a lookup table (not multiplicative) we can have a larger std to allow the model to explore more directions without risking exploding gradients.
        torch.nn.init.normal_(self.lm_head,mean=0.0, std=0.001) # This layer is the last one, that converts the hidden layers to logits. The std is low, so as to have the 
                                                                # init weights close to 0, so that we don't spike up the loss early in the training (because, most of the words' probability from the vocab will be 0 when it predicts the next token.)


        n_embed=self.config.n_embed
        s=3**0.5*n_embed**-0.5 #We use sqrt(3) to ensure uniform achieves the same std as Normal. (because 99.8% of the data in a normal dist. lies in [-3,3] ).


        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s,s) # We initialize using uniform, to avoid large outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s,s)
            torch.nn.init.uniform_(block.attn.c_v.weight,-s,s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) #We initialize it with zeros initially so that they behave as identity functions in the residual stream initially and gradually deepen the network
            torch.nn.init.uniform_(block.mlp.c_fc.weight,-s,s) #c_fc=fully connected
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight,-s,s) # The initial weights are completely over written by the uniform dist. i.e the value in the tensor is randomly chosen between [-s,s].

        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)


        self.resid_lambda.fill_(1.0)    #Typical residual connection at init.
        self.x0_lambdas.fill_(0.0)      # Skip Connection to input is disabled at init. 




