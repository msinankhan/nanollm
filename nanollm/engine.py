import torch
import torch.nn.functional as F
import signal
import warnings

from contextlib import contextmanager
from collections import deque
from nanollm.commons import compute_init, autodetect_device_type, COMPUTE_DTYPE
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
        assert self.get_pos()==0, f"Cannot fill a non-empty cache"
        assert self.n_layers==other.n_layers and self.n_heads==other.n_heads and self.head_dim==other.head_dim
        assert self.max_seq_len>=other.max_seq_len

        other_pos=other.get_pos()

        self.k_cache[:,:,:other_pos,:,:]= other.k_cache[:,:,:other_pos,:,:]
        self.v_cache[:,:,:other_pos,:,:] = other.v_cache[:,:,:other_pos,:,:]
        self.cache_seqlens.fill_(other_pos)


@torch.inference_mode()
def sample_next_token(logits,rng,temperature=1.0,top_k=None):
    assert temperature>=0.0, f"Temperature has to be non-negative:{temperature}"

    if temperature==0.0:
        return torch.argmax(logits,dim=-1,keepdim=True)
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
    


class RowState: 
    """
    Each row: Independent generation trajectory.
    With each prompt, we want n_sample generations.
    Each Sample is its own evolving sequence.
     
    Each row helps tell the engine:
     (i)   Are we inside a python tool block?
     (ii)  Do we need to force inject tokens?
     (iii) Is this sample already finished?
     (iv)  What tokens belong to tool expression?
      
     """
    def __init__(self,current_tokens):
        self.current_tokens=current_tokens or [] #Stores the entire token sequence so far.
        self.forced_tokens=deque() # Used during tool call, when the engine must override sampling and outputs must be injected verbatim.
        self.in_python_block=False # Helps to identify tokens that belong to a python exp. It is turned to True when the appropriate tag is detected.
        self.python_expr_tokens=[] # Raw token IDs inside a python block which will be used to decode and run the code exp.
        self.completed=False # Helps to identify if a given row is completed.


class Engine:
    def __init__(self,model,tokenizer):
        self.model=model
        self.tokenizer=tokenizer # This is needed for tool use. 
         

    @torch.inference_mode()
    def generate(self, tokens,num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens,list) and isinstance(tokens[0],int), "Expecting a list of int."
        device=self.model.get_device()

        dtype=COMPUTE_DTYPE
        rng=torch.Generator(device=device)
        rng.manual_seed(seed)


        get_special= lambda s: self.tokenizer.encode_special(s)
        python_start=get_special("<|python_start|>")
        python_end=get_special("<|python_end|>")
        output_start=get_special("<|output_start|>")
        output_end=get_special("<|output_end|>")
        assistant_end=get_special("<|assistant_end|>")
        bos=self.tokenizer.get_bos_token_id()


        #We first run the prompt to get the K & V from it and then we try to expand it by the number of rows as it is identical across the rows.
        m=self.model.config
        kv_model_kwargs={"num_heads":m.n_kv_head,"head_dim":m.n_embed//m.n_head, "num_layers":m.n_layer}
        kv_cache_prefill=KVCache(
            batch_size=1, # First, generate only one sample for the next token
            seq_len=len(tokens), 
            device=device, dtype=dtype,
            **kv_model_kwargs
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits=self.model.forward(ids,kv_cache=kv_cache_prefill)

        logits=logits[:,-1,:].expand(num_samples,-1) # The shape of logits is (B,T,V) in this case (1,T,V). We only need the last token preds, hence (:,-1,:) and we expand it to num_samples 
                                                     # Expand simply broadcasts the same tensor along the.    

        kv_length_hint=(len(tokens)+max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode=KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs
        )

        kv_cache_decode.prefill(kv_cache_prefill) # This copies the KV cache
        del kv_cache_prefill

        row_states=[RowState(tokens.copy()) for _ in range(num_samples)]

        num_generated=0
        while True:
            if max_tokens is not None and num_generated>=max_tokens:
                break

            if all(state.completed for state in row_states):
                break

            next_ids=sample_next_token(logits,rng,temperature,top_k)
            sampled_tokens=next_ids[:,0].tolist()


            token_column=[] #Contains next token id along each row.
            token_masks=[] #Contains a mask to tell wether it was a generated token(1) or a forced token(0)

            for i,state in enumerate(row_states):
                is_forced=len(state.forced_tokens)>0
                token_masks.append(0 if is_forced else 1)
                next_token=state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)

                state.current_tokens.append(next_token) #Update the row with the new token.

                if next_token==bos or next_token==assistant_end:
                    state.completed=True


                if next_token == python_start:
                    state.in_python_block=True
                    state.python_expr_tokens=[]
                elif next_token==python_end and state.in_python_block:
                    state.in_python_block=False
                    if state.python_expr_tokens:
                        expr=self.tokenizer.decode(state.python_expr_tokens)
                        result=use_calculator(expr)

                        if result is not None:
                            result_tokens=self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)

                    state.python_expr_tokens=[]

                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)


            yield token_column, token_masks
            num_generated+=1

            #Prepare logits for next iteration. 
            ids=torch.tensor(token_column,dtype=torch.long, device=device).unsqueeze(1)
            logits=self.model.forward(ids, kv_cache=kv_cache_decode)[:,-1,:]   #(B, Vocab_size)


    def generate_batch(self,tokens,num_samples=1,**kwargs):
        assistant_end=self.tokenizer.encode_special("<|assistant_end|>")
        bos=self.tokenizer.get_bos_token_id()
        results=[tokens.copy() for _ in range(num_samples)]
        masks=[[0]*len(tokens) for _ in range(num_samples)] #Copy the same prompt across all rows (num_samples).


        completed=[False] *num_samples

        for token_column, token_masks in self.generate(tokens,num_samples,**kwargs):
            for i , (token,mask) in enumerate(zip(token_column,token_masks)):
                if not completed[i]:
                    if token==bos or token == assistant_end:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            
            if all(completed):
                break

        return results, masks
    

if __name__=="__main__":

    import time
    device_type=autodetect_device_type()
    ddp,ddp_rank,ddp_local_rank,ddp_world_size,device=compute_init(device_type)

    model,tokenizer,meta=load_model("base",device,phase="eval")
    bos_token_id=tokenizer.get_bos_token_id()

    kwargs=dict(max_tokens=64,temperature=0.0)
    prompt_tokens=tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)

    synchronize=torch.cuda.synchronize if device_type=="cuda" else lambda:None

    generated_tokens=[]
    synchronize()
    t0=time.time()
    stream=model.generate(prompt_tokens,**kwargs)

    for token in stream:
        generated_tokens.append(token)
        chunk=tokenizer.decode([token])
        print(chunk,end="",flush=True)

    print()
    synchronize()
    t1=time.time()

    print(f"Reference time: {t1-t0:.2f}s")
    reference_ids=generated_tokens

    generated_tokens=[]
    engine=Engine(model,tokenizer)

    stream=engine.generate(prompt_tokens,num_samples=1,**kwargs)
    synchronize()
    t0=time.time()

    for token_column,token_masks in stream:
        token=token_column[0]
        generated_tokens.append(token)
        chunk=tokenizer.decode([token])
        print(chunk,end="", flush=True)

    print()
    synchronize()
    t1=time.time()
    print(f"Engine time: {t1-t0:.2f}s")

    for i in range(min(len(reference_ids),len(generated_tokens))):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")





