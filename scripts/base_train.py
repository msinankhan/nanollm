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

from nanollm.gpt import GPT, GPTConfig
from nanollm.dataloader import tokenizing_distributed_data_loader_with_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanollm.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
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
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config=vars(args).copy()


device_type= autodetect_device_type() if args.device_type=="" else args.device_type
ddp,ddp_rank,ddp_local_rank,ddp_world_size,device=compute_init(device_type)
master_process=ddp_rank==0

autocast_ctx= torch.amp.autocast(device_type=device_type,dtype=torch.bfloat16) if device_type=="cuda" else nullcontext()
synchronize= torch.cuda.synchronize if device=="cuda" else lambda:None #This blocks CPU until GPU finishes all queued work. 
                                                                       # This helps us in measuring accurately the GPU compute time. 
                                                                       #Eg., when loss.backward() is called, the CPU queues the kernel launch 
                                                                       #and GPU executes it later CPU moves ahead. 
                                                                       #When we call time.time() we are measuing the CPU time, 
                                                                       #unless we block CPU until GPU finishes all the queued work with cuda.synchronize()
get_max_memory= torch.cuda.get_max_memory if device=="cuda" else lambda:0


if device_type =="cuda":
    gpu_device_name=  torch.cuda.get_device_name(0)
    gpu_peak_flops=get_peak_flops(gpu_device_name)

    print0(f"GPU: {gpu_device_name}| GPU PEAK FLOPs (BF16): {gpu_peak_flops:.2e}")

else:
    gpu_peak_flops=float('inf')



use_dummy_wandb = args.run =="dummy" or not master_process
wandb_run=DummyWandb() if use_dummy_wandb else wandb.init(project="nanollm", name=args.run,config=user_config)

if HAS_FA3:
    print0("Using FA3.")
else:
    print0("!"*80)
    print0("Warning: FA3 is NOT available, using Pytorch SDPA fallback")

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
        sequence=args.seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads,
        n_embd=model_dim, window_pattern=args.window_pattern
    )
    with torch.device("meta"):
        model_meta=GPT(config)
    return model_meta


model =build_model_meta(args.depth) # 1) Build the model on meta data.
model_config=model.config
model_config_kwargs=asdict(model_config)
print0(f"Model Config: \n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) #2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized


base_dir=get_base_dir()
output_dirname=args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir= os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1

if resuming:
    print0(f"Resuming optimization from step: {args.resume_from_step}")
    model_data, optimizer_data, meta_data= load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data


if args.fp8:
    if device != "cuda":
        print0(f"fp8 requires CUDA, ignoring --fp8 flag.")
    else:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        def fp8_module_filter(mod:nn.Module, fqn:str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features%16 or mod.out_features %16 !=0:
                return False
            return True
        
        fp8_config=Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model,config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers=sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped=sum(1 for m in model.modules() if isinstance(m,nn.Linear)) - num_fp8_layers
        print0(f" FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)")


