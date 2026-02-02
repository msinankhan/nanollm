import torch
import torch.nn.functional as F
import signal
import warnings

from contextlib import contextmanager
from collections import deque
from nanollm.commons import compute_init, autodetect_device_type
from nanollm.checkpoint_manager import load_model
from contextlib import nullcontext


@contextmanager
def timeout(duration,formula): 
    """
    This wraps execution in a wall-clock timeout enforced by the OS.
    Python’s eval() can hang (e.g. infinite loops, massive computation).
    We want hard time limits.
    """
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration}s")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula,max_time=3):
    try:
        with timeout(max_time,formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",SyntaxWarning)
                return eval(formula, {"__builtins__":{}},{}) # Eval basically takes a string and executes it as a Python expression. Ex.: eval("1 + 2 * 3") = 7
                                                             # eval(expr, globals, locals) is eval's signature. By making globals_dict["__builtins__"]= {}, we are overriding
                                                             #the real python builtins. In this universe: No functions exist; No imports exist ; No IO exists
                                                             # locals = {} → empty namespace

    except Exception as e:
        signal.alarm(0) #Cancels any pending alarm.
        return None
    

def use_calculator(exp):
    exp=exp.replace(",","")

    if all([x in "0123456789*+-/.() " for x in exp]):
        if "**" in exp:
            return None
        return eval_with_timeout(exp)
    
    allowed_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in exp]):
        return None
    
    dangerous_expressions=['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    
    exp_lower=exp.lower()

    if any(pattern in exp_lower for pattern in dangerous_expressions):
        return None
    
    if '.count(' not in exp:
        return None
    
    return eval_with_timeout(exp)


class KVCache:

    def __init__(self,batch_size,num_heads,seq_len, head_dim,num_layers,device, dtype):
        self.batch_size=batch_size
        self.max_seq_len=seq_len
        self.n_layers=num_layers
        self.n_heads=num_heads
        self.head_dim=head_dim

        self.k_cache=torch.zeros(num_layers,batch_size,seq_len,num_heads,head_dim,device=device,dtype=dtype)
        self.v_cache=torch.zeros(num_layers,batch_size,seq_len,num_heads,head_dim,device=device,dtype=dtype)


        self.cache_seqlens=torch.zeros(batch_size,device=device,dtype=torch.int32) #Current seq lenght. 
                                                                                  # Helps model know which part of the cache is valid.
                                                                                  #K/V values can be zero legitimately , hence this helps in identifying 

    def reset(self):
        self.cache_seqlens.zero_() #Resets the cache to 0.

    def get_pos(self):
        return self.cache_seqlens[0].item() #item() returns the tensor as a python number
    
    def get_layer_cache(self,layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def advance(self,num_tokens):
        self.cache_seqlens+=num_tokens #Advances the cache by num_tokens


    def prefill(self,other):

        """Copies KV from another cache into this one.
        Will be used when we want to generate multiple samples in parallel. """
        assert self.get_pos==0, f"Cannot fill a non-empty cache"
        assert self.n_layers==other.n_layers and self.n_heads==other.n_heads and self.head_dim==self.head_dim
        assert self.max_seq_len>=other.max_seq_len

        other_pos=other.get_pos()

        self.k_cache[:,:,:other_pos,:,:]= other.k_cache[:,:,:other_pos,:,:]
        self.v_cache[:,:,:other_pos,:,:] = other.v_cache[:,:,:other_pos,:,:]
        self.cache_seqlens.fill_(other_pos)


@torch.inference_mode()
def sample_next_token(logits,rng,temperature=1.0,top_k=None):
    assert temperature>=0.0, f"Temperature has to be non-negative:{temperature}"

    if temperature==0:
        return torch.argmax(logits,dims=-1,keepdim=True)
    if top_k is not None and top_k>0:
        k=min(top_k,logits.size(-1))
        vals,idx=torch.topk(logits,k,dim=-1)
        vals=vals/temperature #Temperature rescales relative differences, higher_temp-> flatter distribution; lower_temp->peakier distribution.
        probs=F.softmax(vals,dim=-1)
        choice=torch.multinomial(probs,num_samples=1, generator=rng)
        return idx.gather(1,choice) #choice is the index in the top_k list, but we need the original token_id.
    else:
        logits=logits/temperature
        probs=F.softmax(logits,dim=-1)
        return torch.multinomial(probs,num_samples=1,generator=rng)
    


