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

    n_embed : int = 768 # Size of the residual stream, Everything lives in this space: embeddings, attention outputs, MLP outputs. This is the single most important dimension in the entire model.

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

        self.resid_lambda.fill_(1.0)    #Typical residual connection at init.
        self.x0_lambdas.fill_(0.0)      # Skip Connection to input is disabled at init. 

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight,-s,s) # The initial weights are completely over written by the uniform dist. i.e the value in the tensor is randomly chosen between [-s,s].

        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        head_dim=self.config.n_embed//self.config.n_head
        cos,sin=self._precompute_rotary_embeddings(self.rotate_seq_len,head_dim)
        self.cos,self.sin=cos,sin

        if self.transformer.wte.weight.device.type =="cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self,seq_len,head_dim, base=10000,device=None):
        """We calculate the cos and sin for every position in the seq_len and store it in the cache to use it later.
           Here, we are basically caluclating m.(θi) first and then use it to calculate cos and sin for every position."""

        if device is None:
            device= self.transformer.wte.weight.device

        channel_range=torch.arange(0,head_dim,2,dtype=torch.float32,device=device) # This gives you an tensor like [0,2,4...head_dim-2]
        inv_freq=1.0/(base**(channel_range/head_dim)) # This gives you θ i = b ^ (− 2 i / d) the 2i comes from channel_range, as when i=0,1,2, we have [0,2,4...]
        t=torch.arange(seq_len,dtype=torch.float32,device=device) 

        freqs=torch.outer(t,inv_freq) #This is the angle m.θi for position m and dimension pair i.

        cos,sin=freqs.cos(), freqs.sin() # The rows here refer to positions m and columns refer to θ i
        cos,sin=cos.bfloat16,sin.bfloat16

        cos,sin=cos[None,:,None,:], sin[None,:,None,:] #Final shape: (1, seq_len, 1, head_dim/2). This allows automatic broadcasting over batch sizes and attention heads. Which allows us to do q * cos without reshaping. 
        return cos,sin #The function returns these tensors, which are registered as buffers in the model.
    


    def _compute_window_sizes(self,config):
        """PaLM style optimization.
           returns (left,right), left=-1 for full context. """
        pattern=config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid pattern {pattern} use S and L only."

        long_window=config.seq_len
        short_window=long_window//2

        char_to_window={
            "L" : (long_window,0),
            "S" : (short_window,0)
        }

        window_size=[]

        for layer_idx in range(config.n_layer):
            char=pattern[layer_idx%len(pattern)]
            window_size.append(char_to_window[char])

        window_size[-1]=(long_window,0) # Final layer gets full context. 

        return window_size

    def get_device(self):
        return self.transformer.wte.weight.device
    

    def estimate_flops(self):
        """This returns the number of Floating Point Operations, we only consider matmul operations and ignore scalar addition and multiplication."""

        nparams=sum(p.numel() for p in self.parameters) #This gives you the whole set (including non-matmul operations which we gotta substract.)

        value_embeds=sum (ve.weight.numel() for ve in self.value_embeds.values())

        nparams_exclude=(self.transformer.wte.weight.numel() + value_embeds    # Get rid of embeddings as they are just look ups
                         + self.resid_lambda.numel()+self.x0_lambdas.numel())  # Get rid of non-matmul ops as well.


        h,q,t=self.config.n_head, self.config.n_embed//self.config.n_head, self.config.sequence_len

        attn_flops=0

        for window_size in self.window_sizes: #Attn isn't captured in parameters because there is weight matrix involved.
            window=window_size[0]             #As in, q=X.W_q (W_q is wt. matrix), where as in attn (q.k^T), we don't have a weight matrix and won't be captured in parameter count.
            effective_seq=t if window<0 else min(window,t)
            attn_flops+=12*h*q*effective_seq #Because there are 12 heads(1/layer).

        num_flops_per_token=6*(nparams-nparams_exclude) + attn_flops # Each token contributes to about 6 Floating Point Operations, (2 forward prop. + 4 in backward prop.)
        return num_flops_per_token
    

    def num_sclaing_params(self):
        nparams=sum(p.numel() for p in self.parameters())
        return nparams


    def setup_optimizers(self,
                         unembedding_lr=0.004, # This is for the logits, and it is extremely sensitive, hence a very low lr
                         embedding_lr=0.2,     # The embedding space is huge, hence a larger lr.
                         matrix_lr=0.02,       # Momentum Based.
                         weight_decay=0.0,
                         adam_betas=(0.8,0.95),
                         scalar_lr=0.5):
        

        model_dim=self.config.n_embed
        ddp,rank,local_rank,world_size=get_dist_info()


        matrix_params=list(self.transformer.h.parameters())
        value_embed_params=list(self.value_embeds.parameters())
        embedding_params=list(self.transformer.wte.parameters())
        lm_head_params=list(self.lm_head.parameters())
        resid_lambda_params=[self.resid_lambda]
        x0_params=[self.x0_lambdas]

        assert len(list(self.parameters())) == len(matrix_params) + len(value_embed_params) +len(embedding_params) +len(lm_head_params) + len(resid_lambda_params) + len(x0_params)


        # Now we create AdamW Optimizer for embedding, lm_head and per-layer scalars.

        dmodel_lr_scale=(model_dim/768) ** -0.5 # We scale the LR by  ∝ 1/√dmodel ( As the LR is tuned for 786 dim model through experiments and we have now use a scaled version of it to our model size)

        print0(f"Scaling the LR for AdamW parameters by  ∝ 1/√{model_dim}/768 = {dmodel_lr_scale:6.f}")

        adam_groups=[
            dict(params=lm_head_params, lr=unembedding_lr*dmodel_lr_scale),
            dict(params=embedding_params,lr=embedding_lr*dmodel_lr_scale),
            dict(params=value_embed_params,lr=embedding_lr*dmodel_lr_scale),
            dict(params=resid_lambda_params, lr=scalar_lr*0.01), # These are very sensitive as they accumulate in the residual stream.
            dict(params=x0_params,lr=scalar_lr)
        ]

        adamw_kwargs=dict(betas=adam_betas, eps=1e-10,weight_decay=0.0) # Weight decay is only for the Muon Optimizer.
        AdamFactory=DistAdam if ddp else partial(torch.optim.Adam,fused=True)
        adamw_optimizer=AdamFactory(adam_groups,**adamw_kwargs)


        muon_kwargs=dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory=DistMuon if ddp else Muon
        muon_optimizer=MuonFactory(matrix_params,**muon_kwargs)

        optimizers=[adamw_optimizer,muon_optimizer]

        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"]= group["lr"]

        return optimizers
    
    def forward(self,idx,targets=None,kv_cache=None,loss_reduction='mean'):
        B,T=idx.size()

        assert T<self.cos.size(1), f"Sequence Length grew beyond the rotary cache: {T}>{self.cos.size(1)}"
        assert idx.device==self.cos.device, f"Rotary Embeddings and the idx are on different devices: {idx.device}!= {self.cos.device}"
        assert self.cos.dtype==torch.bfloat16, f"Rotary embeddings must be in bfloat: {self.cos.dtype}"

        # We now have to grab the rotary embeddings for the current sequence.
        T0=0 if kv_cache is None else kv_cache.get_pos()
        cos_sin=self.cos[:,T0:T0+T],self.sin[:,T0:T0+T] # The first ":" is ignored, it refers to the Batch dimension, as the shape of returned by _precompute_rotary_embeddings returns a shape (1,seq_len,1,head_dim/2) -> (B,T,H,Head_dim/2) 


        x=self.transformer.wte(idx)
        x=norm(x)
        x0=x
        for i,block in enumerate(self.transformer.h):
            x=self.resid_lambda[i]*x +self.x0_lambdas[i]*x0
            ve=self.value_embeds[str(i)] if str(i) in self.value_embeds else None
            x=Block(x,ve,cos_sin,self.window_sizes[i], kv_cache)
        
        x=norm(x)


        softcap=15
        logits=self.lm_head(x)
        logits=[...,self.config.vocab_size]
        logits=logits.float()

        logits=softcap* torch.tanh(logits/softcap)

        if targets is not None:
            # Cross entropy wants (N,C), N= num. of independent predictions, C= number of classes (vocab_size). But the input here is (B,T,V)
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)), # This turns (B,T,V) -> (B*T,V). Think of "-1" in view as x. Now we want to rearrange the data such that, x * V will be return all the elements which it had previously, hence we get (B*T,V)
                                 targets.view(-1), # This flattens it, i.e (B,T) -> (B*T). 
                                 ignore_index=-1, # We ignore the last index after the last token is predicted. 
                                 reduction=loss_reduction 
                                )
            return loss
        else:
            return logits




    @torch.inference_mode() #disables auto grad
    def generate(self,tokens,max_tokens,temperature=1.0,top_k=None, seed=42):
        assert isinstance(tokens,list) 

        device=self.get_device()
        if temperature>0:
            rng=torch.Generator(device=device)
            rng.manual_seed(seed)
        ids=torch.tensor([tokens],dtype=torch.long, device=device) #torch.long=64bit signed integer equivalent to torch.int64

        for _ in range(max_tokens):
            logits=self.forward(ids) # It recieves (B,T,Vocab_size) meaning, for every position in the sequence the model predicts the next token after that position (even for the tokens who's next positions we have defined).

            logits=logits[:,-1,:] # We only need the prediction of after the last token. (B,Vocab_size)

            if top_k is not None:
                v,_=torch.topk(logits, min(top_k,logits.size(-1)))
                logits[logits[:,v[-1]]]=-float('inf') # We are setting the logits of all the tokens beyong the kth largest logit to -inf to make sure they will not be sampled at all.

            if temperature>0:
                logits=logits/temperature #Dividing by temperature scales entropy.
                probs=F.softmax(logits, dim=-1) # dim =-1 because (B,V), we have to choose from Vocab only. 
                next_id=torch.multimodal(probs,dim=-1,num_samples=1,generator=rng) #torch.multimodal chooses a number from the uniform probability distribution. The generator helps in choosing the number.
            else:
                next_id=torch.argmax(logits,dim=-1,keepdim=True) 

            ids=torch.cat((ids,next_id),dim=1) 
            token=ids.item()
            yield token


            

