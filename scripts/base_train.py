import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" # Changes Pytorch's CUDA memory allocator behaviour, allowing CUDA memory blocks to expand dynamically.
import gc 
import json
import time 
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager
import wandb
import torch
import torch.distributed as dist

from nanollm.gpt import GPT, GPTConfig
from nanollm.dataloader import tokenizing_distributed_data_loader_with_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanollm.commons import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
from nanollm.tokenizer import get_tokenizer, get_token_bytes    
from nanollm.checkpoint_manager import save_checkpoint, load_checkpoint
from nanollm.loss_eval import evaluate_bpb
from nanollm.engine import Engine
from scripts.base_eval import evaluate_core

print_banner()


#------------------------------------------------------------------------------------------------------------------------------------

def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
if HAS_FA3:
    print0(" HAS_FA3: True")
else:
    print0(" HAS_FA3: False")

_override_impl = None

def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


#--------------------------------------------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description="Pretrain base model")

# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="L", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-params-data-ratio", type=float, default=20, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")

# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
# parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
# parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
# parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
# parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=-1, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=250, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config=vars(args).copy()


device_type= autodetect_device_type() if args.device_type=="" else args.device_type
ddp,ddp_rank,ddp_local_rank,ddp_world_size,device=compute_init(device_type)
master_process=ddp_rank==0

# autocast_ctx= torch.amp.autocast(device_type=device_type,dtype=torch.bfloat16) if device_type=="cuda" else nullcontext()
synchronize= torch.cuda.synchronize if device_type=="cuda" else lambda:None #This blocks CPU until GPU finishes all queued work.
                                                                       # This helps us in measuring accurately the GPU compute time. 
                                                                       #Eg., when loss.backward() is called, the CPU queues the kernel launch 
                                                                       #and GPU executes it later CPU moves ahead. 
                                                                       #When we call time.time() we are measuing the CPU time, 
                                                                       #unless we block CPU until GPU finishes all the queued work with cuda.synchronize()
get_max_memory= torch.cuda.max_memory_allocated if device_type=="cuda" else lambda:0


if device_type =="cuda":
    gpu_device_name=  torch.cuda.get_device_name(0)
    gpu_peak_flops=get_peak_flops(gpu_device_name)

    print0(f"GPU: {gpu_device_name}| GPU PEAK FLOPs (BF16): {gpu_peak_flops:.2e}")

else:
    gpu_peak_flops=float('inf')

print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")


use_dummy_wandb = args.run =="dummy" or not master_process
wandb_run=DummyWandb() if use_dummy_wandb else wandb.init(project="nanollm", name=args.run,config=user_config)

# if HAS_FA3:
#     print0("Using FA3.")
# else:
#     print0("!"*80)
#     print0("Warning: FA3 is NOT available, using Pytorch SDPA fallback")

#     if args.window_pattern != "L":
#         print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
#         print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
#     print0("!" * 80)


using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)


tokenizer=get_tokenizer()
token_bytes=get_token_bytes(device=device)
vocab_size=tokenizer.get_vocab_size()
print0(f"Vocab Size: {vocab_size:,}")

def build_model_meta(depth):

    base_dim=depth* args.aspect_ratio
    model_dim=((base_dim+args.head_dim-1)//args.head_dim) * args.head_dim # This is the size of the vector that represents each token inside the transformer.
                                                                          # This is the width of the model.
                                                                          #You can think of it as:
                                                                            # The dimensionality of the representation space
                                                                            # The bandwidth of information per token
                                                                            # The size of the "feature vector" for each token


    num_heads=model_dim//args.head_dim         # Inside multi-head attention, we split the representation into multiple heads. 
                                               # IF model_dim = 768 ; num_heads = 12.
                                               # Then: each head works on a (768/12=64) 64-dimensional subspace.



    config= GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads,
        n_embed=model_dim, window_pattern=args.window_pattern
    )
    with torch.device("meta"):
        model_meta=GPT(config)
    return model_meta


model = build_model_meta(args.depth) # 1) Build the model on meta data.
model_config=model.config
model_config_kwargs=asdict(model_config)
print0(f"Model Config: \n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) #2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized


base_dir=get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1

if resuming:
    print0(f"Resuming optimization from step: {args.resume_from_step}")
    model_data, optimizer_data, meta_data= load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data


if args.fp8:
    if device_type != "cuda":
        print0(f"fp8 requires CUDA, ignoring --fp8 flag.")
    else:
        from nanollm.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        def fp8_module_filter(mod:nn.Module, fqn:str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features%16 or mod.out_features %16 !=0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True
        
        fp8_config=Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model,config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers=sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped=sum(1 for m in model.modules() if isinstance(m,nn.Linear)) - num_fp8_layers
        print0(f" FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)")


        
@contextmanager #Transforms a generator function into a context manager.
def disable_fp8(model):
    import torch.nn as nn
    fp8_locations=[]   # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name,attr_name=name.rsplit('.',1)
                parent=module.get_submodule(parent_name)

            else:
                parent=model #If module name is: transformer.blocks.3.attn.proj
                             #Then: parent_name = transformer.blocks.3.attn, attr_name = proj
                attr_name=name

            fp8_locations.append((parent,attr_name,module))


    if not fp8_locations:
        yield
        return


    for parent, attr_name,fp8_module in fp8_locations:
        linear=nn.Linear(                                # Regular linear layer with same:
            fp8_module.in_features,                         # input dim
            fp8_module.out_features,                        # output dim
            bias=fp8_module.bias is not None,               
            device= "meta",
            dtype=fp8_module.weight.dtype     #dtype remains bf16. Because The stored master weights remain in BF16 (or FP16).
        )

        linear.weight=fp8_module.weight      # They are NOT copying weights. They are sharing the same tensor. 

        if fp8_module.bias is not None:
            linear.bias=fp8_module.bias

        setattr(parent, attr_name, linear)


    try:
        yield

    finally:
        for parent,attr_name,fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)



orig_model=model                            #Raw PyTorch model (used for evaluation, checkpointing, generation)
model=torch.compile(model,dynamic=False)     #torch.compile() (introduced in PyTorch 2.0) wraps your model in a compiled graph module.
                                                #  After model = torch.compile(model)
                                                # model is no longer your raw nn.Module
                                                

                                            #That wrapper:

                                                # Traces the model
                                                # Freezes parts of the structure
                                                # Guards assumptions
                                                # Caches kernels
                                                # Removes Python overhead
                                                # Optimizes memory + compute scheduling

                                            # dynamic=False because:
                                                #In training, the shapes are constant throughout: x shape = [batch_size, seq_len]; y shape = [batch_size, seq_len]


params_count=model.num_sclaing_params()
print0("Parameter Counts:")
for key,value in params_count.items():
    print0(f"{key:24s}:{value:,}")
num_params=params_count['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")


## 1) Use scaling laws to determine the optimal training horizon in tokens

def get_scaling_params(m):
    params_count=m.num_sclaing_params()
    scaling_params=params_count['transformer_matrices'] + params_count['lm_head']
    return scaling_params

num_sclaing_params = get_scaling_params(model)
target_tokens=int(args.target_params_data_ratio * num_sclaing_params)



d12_ref=build_model_meta(12)
D_REF=args.target_params_data_ratio* get_scaling_params(d12_ref)    #how many tokens the reference model should train on according to scaling laws.
B_REF= 2**19


#2) With the token horizons, we calculate the optimal batch sizes. 
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.

#The Real Principle: μP Scaling
#When you change model size:
    # depth ↑
    # width ↑
    # parameters ↑

#  The gradient magnitudes change.
# Which means:
    # the same learning rate no longer works
    # the optimizer behaves differently
    # training may diverge or slow down

# So we need a way to scale hyperparameters so that:    
    # training behavior stays the same
    # regardless of model size.


#The reference model is the model where hyperparameters are defined.
# Now we train a bigger model, instead of re-tuning everything, we compute:
    # param_ratio = new_model_scaling_params / reference_scaling_params

# μP theory says:
    # If we scale parameters correctly relative to a reference model, then training curves remain invariant
    # Meaning:
        # Loss vs tokens looks almost identical across model sizes.


total_batch_size=args.total_batch_size
if total_batch_size==-1:
    batch_size_ratio= target_tokens/D_REF
    predicted_batch_size= B_REF* batch_size_ratio **0.383   #As you train on more tokens, the optimal batch size should increase. But not linearly.So optimal scaling is roughly: B ∝ D^0.383 which is sublinear growth.
                                                            #If you don't increase batch size
                                                                #1) Training becomes inefficient
                                                                #2) Gradient noise dominates updates.
                                                            #If you increase batch size too fast, you get:
                                                                #optimization slowdown
                                                                #poor generalization
                                                            # The 0.383 exponent is empirically close to optimal.


                                                            #What Happens if We Scale Linearly (B ∝ D)

                                                                # Suppose training tokens increase 10×:

                                                                    # D_new = 10 × D_ref

                                                                # If we scaled batch size linearly:

                                                                    # B_new = 10 × B_ref

                                                                # Now look at the number of parameter updates:

                                                                # updates = total_tokens / batch_size

                                                                # So:

                                                                    # updates_new = (10D) / (10B) = D/B

                                                                 # Meaning:

                                                                    # the number of optimizer steps stays the same
                                                                    # Why This Is Bad

                                                                    # Even though we train on 10× more data, we perform no additional learning steps.

                                                                # So the model:

                                                                    # sees more tokens

                                                                    # but performs the same number of updates

                                                                 # This causes:

                                                                    # under-optimization

                                                                    # The model cannot properly absorb the additional data.



                                                                #Batch size controls gradient variance.

                                                                    # Large batch:

                                                                        # low gradient noise

                                                                    # Small batch:

                                                                        # high gradient noise

                                                                    # If we scale batch size too aggressively:

                                                                        # gradient noise → near zero

                                                                    # Then SGD behaves like deterministic gradient descent, which is known to:

                                                                        # converge slower

                                                                        # generalize worse

                                                                    # So linear scaling would:

                                                                        # reduce useful stochasticity
    total_batch_size= 2**round(math.log2(predicted_batch_size))
    print0(f"Auto-complete optimal batch size: {total_batch_size:,} tokens")


batch_lr_scale=1.0 # If batch_size = reference batch size, the LR should remain the same (because matrix_lr=args.matrix_lr * batch_lr_scale)
batch_ratio=total_batch_size/B_REF
if batch_ratio!=1.0:
    batch_lr_scale = batch_ratio ** 0.5 # We use larger LR for a larger batch, as a gradient computed from a larger batch is less noisy. ; 
                                        #The gradient we compute is an estimate of the true gradient:
                                            
                                        # 𝑔 = ∇𝐿(𝜃) + 𝜖 
                                            
                                            
                                        # where:    
                                            
                                        # ∇𝐿(𝜃)= true gradient  
                                            
                                        # 𝜖 = stochastic noise from sampling the batch  
                                            
                                        #Gradient noise's variance scales : as Var(g) ∝ 1/B, so douobling batch size halves the variace allowing us to increase the LR as when B increases, the gradient estimate becomes more stable, optimizer can take larger steps safely. 


                                        # We don't use linear scaling (we do it for SGD) for ADAMW because adam normalizes gradients internally so it needs less aggressive scaling.
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference:{B_REF:,})")



# GOTTA COME BACK TO THIS LATER:
#calculate the appropriate weight decay scaling using the  batch size and the token horizon.

# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.: https://arxiv.org/abs/2405.13698

# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)

weight_decay_scaled=args.weight_decay * math.sqrt(total_batch_size/B_REF) * (D_REF/target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weights from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")


optimizer=model.setup_optimizer(
    # ADAMW hyperparameters
    unembedding_lr=args.unembedding_lr* batch_lr_scale,
    embedding_lr=args.embedding_lr*batch_lr_scale,
    scalar_lr=args.scalar_lr *batch_lr_scale,
    #Muon Hyperparameters:
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay= weight_decay_scaled
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data


# scaler= torch.amp.GradScaler() if COMPUTE_DTYPE ==


dataloader_resume_state_dict= None if not resuming else meta_data["dataloader_state_dict"]
train_loader= tokenizing_distributed_data_loader_with_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split='train', device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda : tokenizing_distributed_data_loader_with_bos_bestfit(tokenizer, args.device_batch_size,args.max_seq_len, split="val", device=device)
x,y, dataloader_state_dict= next(train_loader)

assert args.num_iterations >0 or args.target_params_data_ratio>0 or args.target_flops >0

if args.num_iterations>0:
    num_iterations=args.num_iterations
    print0(f"Using user-provided number of iterations; {num_iterations:,}")
elif args.target_flops>0:
    num_iterations=round(args.target_flops/(num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs {num_iterations:,}")

elif args.target_params_data_ratio >0:
    num_iterations = target_tokens //total_batch_size
    print0(f"Calculated number of iterations from target data: param ratio: {num_iterations:,}")

else:
    raise ValueError("No training horizaon specified.")

total_tokens= total_batch_size * num_iterations # Actual number of tokens we train on.
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens: Scaling params ratio: {total_batch_size* num_iterations/num_sclaing_params:.2f}")
print0(f"Total training FLOPs estimate  {num_flops_per_token * total_tokens:e}")

                                
def get_lr_multiplier(it):
    warmp_iters = args.warmup_steps
    warmdown_iters= round(args.warmdown_ratio * num_iterations)
    if it< warmp_iters:
        return (it +1)/ warmp_iters
    elif it <=num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations -it) /warmdown_iters
        return progress *1.0 + (1-progress) * args.final_lr_frac
    

def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


def get_weight_decay(it):
    return weight_decay_scaled * 0.5 *(1+math.cos(math.pi * it/num_iterations))  #Cosine decay to zero over the course of training.


if not resuming:
    step=0
    val_bpb = None 
    min_val_bpb = float('inf')
    smooth_train_loss =0
    total_training_time =0
else:
    step=meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb=meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Here we are trying to figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step.
tokens_per_fwdbwd= args.device_batch_size * args.max_seq_len #tokens per iteration for a single rank. 
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd ==0
grad_accum_steps= total_batch_size //world_tokens_per_fwdbwd
print0(f"Tokens/micro-batch/rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens/micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size: {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")


while True:
    last_step= step==num_iterations 
    flops_so_far = num_flops_per_token * total_batch_size *step

    if args.eval_every >0 and (last_step or step % args.eval_every==0):
        model.eval()
        val_loader=build_val_loader()
        eval_steps= args.eval_tokens//(args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb= evaluate_bpb(model,val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb : {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step":step,
            "total_training_flops":flops_so_far,
            "total_training_time":total_training_time,
            "val/bpb":val_bpb,
        })
        model.train()

    results={}
    if args.core_metric_every >0 and (last_step or (step>0 and step % args.core_metric_every ==0)):
        model.eval()
        with disable_fp8(orig_model):
            results= evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)

        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step":step,
            "total_training_flops":flops_so_far,
            "core_metric": results['core_metric'],
            "centered_results": results['centered_results']
            })
        model.train()

    if args.sample_every >0 and master_process and (last_step or (step>0 and step % args.sample_every==0)):
        model.eval()
        prompts=[
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x +3 =13, then x is",
            ]
        engine=Engine(orig_model,tokenizer)
        for prompt in prompts:
            tokens=tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _=engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))

        model.train()

    if last_step or (step>0 and step!= args.resume_from_step and args.save_every>0 and step%args.save_every==0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step":step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len":args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state":{
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time":total_training_time,
                },
            },
            rank=ddp_rank,
        )
    if last_step: # termination conditions (TODO: possibly also add loss explosions etc.)
        break

    synchronize() #It forces the CPU to wait until all GPU kernels finish. because GPUs are async and if t1-t0 will be wrong as the CPU continues with the script if we don't synchronize and the time diff is calculated on the CPU.
    t0=time.time()
    for micro_step in range(grad_accum_steps):
        loss=model(x,y)     #The model wants an effective batch size: total_batch_size
                                    #But GPU memory only allows:device_batch_size
                                    #So we simulate a large batch using multiple smaller batches.

                                    # Example:
                                    # total_batch_size = 524k tokens
                                    # micro_batch = 64k tokens
                                    # grad_accum_steps = 8

                                    #Training becomes:
                                    # forward/backward 8 times
                                    # accumulate gradients
                                    # then update weights once

        train_loss=loss.detach() # for logging: We need to detach it from the computation graph to prevent GPU memory leak. detach() creates the same tensor but it doesn't track gradients.

        loss=loss/grad_accum_steps  # each .backward() is a grad sum => normalize loss here

                # if scaler is not None:
                #     scaler.scale(loss).backward()
                # else:
        loss.backward()
        x,y,dataloader_state_dict=next(train_loader) #GPU → running backward pass
                                                             # CPU → loading next data batch

            #End of Micro-Step Loop


            # Step the optimizer:

    lrm=get_lr_multiplier(step)   #This dynamically adjusts: learning rate, optimizer momentum weight decay.
    muon_momentum=get_muon_momentum(step) # Based on training step.
    muon_weight_decay=get_weight_decay(step)


    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

            # if scaler is not None:
            #     scaler.unscale_(optimizer)
            #     # In distributed training, all ranks must agree on whether to skip the step.
            #     # Each rank may independently encounter inf/nan gradients, so we all-reduce
            #     # the found_inf flag (MAX = if any rank found inf, all ranks skip).
            #     if is_ddp_initialized():                                                       #All DDP workers must perform the exact same sequence of optimizer steps. Otherwise the models diverge.
            #         for v in scaler._found_inf_per_device(optimizer).values():                 #Without synchronization:
                                                                                                    #GPU0 → optimizer.step() executed
                                                                                                    #GPU1 → optimizer.step() skipped

                                                                                                #Now the weights become:

                                                                                                    #GPU0 → W_(t+1)
                                                                                                    #GPU1 → W_t

                                                                                                #Now the models no longer match.    
            #             dist.all_reduce(v, op=dist.ReduceOp.MAX)
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f=train_loss.item() #.item() forces GPU sync, because CPU must wait for the value.
    synchronize()
    t1=time.time()
    dt=t1-t0

            

    ema_beta=0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1-ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1-ema_beta**(step+1))
    pct_done=100*step/num_iterations
    token_per_sec=int(total_batch_size/dt)
    flops_per_sec = num_flops_per_token * total_batch_size/dt
    mfu = 100 * flops_per_sec /(gpu_peak_flops * ddp_world_size)

    if step >10:
        total_training_time+=dt

    steps_done = step -10

    if steps_done >0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str=""

    epoch= f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pg_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm:{lrm:.2f} | dt:{dt*1000:.2f}ms | tok/sec: {token_per_sec:,} | bf15_mfu: {mfu:.2f} | epoch: {epoch} | total_time: {total_training_time/60:.2f}m{eta_str}")

    if step %100 ==0:
        log_data={
            "step":step,
            "total_training_flops": flops_so_far,
            "train/loss": debiased_smooth_loss,
            "train/lrm" : lrm,
            "train/dt" : dt,
            "train/tok_per_sec": token_per_sec,
            "train/mfu":mfu,
            "train/epoch":epoch
        }

        wandb_run.log(log_data)


    first_step_of_run = (step==0) or (resuming and step == args.resume_from_step)
    step+=1

    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()

    elif step %5000 ==0:
        gc.collect()

            
print0(f"Peak memory usage: {get_max_memory()/1024/1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")

if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")


from nanollm.report import get_report
get_report().log(section="Base Model Training", data=[
    user_config,
    {
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations":num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens: Scaling params ratio": total_batch_size * num_iterations / num_sclaing_params,
        "DDP World Size" : ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    {
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE Metric Estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time":f"{total_training_time/60:.2f}m",
        "Peak Memory Usage": f"{get_max_memory()/1024/1024:.2f}MiB",
    }
])

wandb_run.finish()
compute_cleanup()
