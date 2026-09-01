import argparse
import json
import time

import torch

from nanollm.commons import (
    compute_init,
    compute_cleanup,
    autodetect_device_type,
    get_peak_bandwidth,
    get_peak_flops,
)
from nanollm.checkpoint_manager import load_model
from nanollm.engine import Engine



def weight_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def bench_generate(engine, prompt_tokens, batch_size, decode_tokens, temperature):

    device = engine.model.get_device()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


    generator = engine.generate(prompt_tokens, num_samples=batch_size,
                                max_tokens=decode_tokens, temperature=temperature)

    t_start = time.perf_counter()
    next(generator)
    torch.cuda.synchronize(device)
    ttft=time.perf_counter() - t_start


    step_times =[]
    while True:
        t0= time.perf_counter()
        try: 
            next(generator)
        except StopIteration:
            break
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

    peak_vram = torch.cuda.max_memory_allocated(device)
    return dict(ttft=ttft, step_times=step_times, peak_vram=peak_vram)


def build_prompt(tokenizer, num_tokens):
    paragraph = ("The history of science is the study of the development of science,"
                 "including both the natural and social sciences. Science is a body of "
                 "empirical, theoretical , and practical knowledge about the natural world. ")

    text = paragraph * max(1, num_tokens // 10)
    tokens = tokenizer.encode(text, prepend = "<|bos|>")
    assert len(tokens) >= num_tokens, "Prompt text too short, increase the repetition"
    return tokens[:num_tokens]


def main():
    parser = argparse.ArgumentParser(description="Inference benchmark" )

    parser.add_argument(
        "-i",
        "--source",
        type=str,
        default="base",
        choices=["base", "mid", "sft", "rl"],
    )
    parser.add_argument("-g", "--model-tag", type=str, default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,32,128",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()


    assert args.prompt_tokens > 0
    assert args.decode_tokens >= 2
    assert args.temperature >= 0

    batch_sizes = [ int(value.strip()) for value in args.batch_sizes.split(",") ]
    assert batch_sizes
    assert all(size > 0 for size in batch_sizes)

    device_type=autodetect_device_type()
    assert device_type == "cuda", "infer_bench currently assumes a CUDA GPU (for timing and VRAM measurement)"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    assert ddp_world_size ==1, "Infer_bench is a single GPU benchmark, run without torchrun"


    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step = args.step)
    config = model.config
    engine= Engine(model,tokenizer)

    max_prompt = config.sequence_len - args.decode_tokens

    prompt_len = min(args.prompt_tokens, max_prompt)
    if prompt_len < args.prompt_tokens:
        print(f"Note: Clamping prompt to {prompt_len} tokens so prompt+decode fits sequence_len= {config.sequence_len}")

    prompt_tokens = build_prompt(tokenizer, prompt_len)

    device_name = torch.cuda.get_device_name(device)
    peak_bw = get_peak_bandwidth(device_name)
    peak_flops = get_peak_flops(device_name)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    w_bytes=weight_bytes(model)
    num_params = sum(p.numel() for p in model.parameters())
    kv_store = model.kv_bytes_per_token()
    context_mid = prompt_len + args.decode_tokens // 2

    kv_read = model.kv_read_bytes(context_mid)

    ceiling_bs1 = peak_bw /(w_bytes + kv_read)

    max_rows = int((total_vram - w_bytes) / (kv_store * config.sequence_len))

    print('=' * 100)
    print(f"Model: {args.source} {meta.get('model_tag', '')} (step {meta['step']}) |"
          f"Depth {config.n_layer}, dim {config.n_embed}, heads {config.n_head}, kv_heads {config.n_kv_head} (GQA)")

    print(f"GPU: {device_name} | peak bandwidth {peak_bw/1e12:.2f} TB/s | peak compute {peak_flops/1e12:.0f} TFLOPS | VRAM {total_vram/2**30:.0f} GiB")
    print("-" * 100)
    dtype_counts ={}

    for p in model.parameters():
        dtype_name=str(p.dtype).replace("torch.", "")
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name,0) + p.numel()

    param_dtypes = ", ".join(f"{n:,} {dtype_name}" for dtype_name, n in sorted(dtype_counts.items()))
    print(f"Parameters: {num_params:,} ({param_dtypes}) | weight bytes as stored: {w_bytes/2**20:.0f} MiB")
    print(f"KV cache: {kv_store:,} bytes/token stored | {kv_read:,} bytes read/step at context {context_mid} "
          f"(window pattern {config.window_pattern})")
    print(f"Theoretical decode ceiling at batch 1: {ceiling_bs1:,.0f} tok/s | "
          f"max ~{max_rows:,} full-context rows in VRAM")
    print("=" * 100)


    payload = {
        "source": args.source,
        "step": meta["step"],
        "model_config": meta["model_config"],
        "gpu": device_name,
        # None (not Infinity) for unknown GPUs, so the last line stays valid JSON
        "peak_bandwidth_bytes_per_sec": peak_bw if peak_bw != float("inf") else None,
        "total_vram_bytes": total_vram,
        "num_params": num_params,
        "param_dtypes": dtype_counts,
        "weight_bytes": w_bytes,
        "kv_bytes_per_token": kv_store,
        "kv_read_bytes_per_step": kv_read,
        "context_mid": context_mid,
        "peak_flops_per_sec": peak_flops if peak_flops != float("inf") else None,
        "decode_flops_per_token": model.estimate_decode_flops(context_mid),
        "ceiling_bs1_tok_per_sec": round(ceiling_bs1, 1) if ceiling_bs1 != float("inf") else None,
        "max_full_context_rows": max_rows,
        "prompt_tokens": prompt_len,
        "decode_tokens": args.decode_tokens,
        "temperature": args.temperature,
        "sweep": [],
    }
