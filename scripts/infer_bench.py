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
    assert max_prompt > 0, ( "decode_tokens must be smaller than sequence_len")

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

    available_vram = max(total_vram - w_bytes, 0)

    max_rows = int(
        available_vram
        / (kv_store * config.sequence_len)
    )

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


    bench_generate(engine, prompt_tokens, batch_size=1, decode_tokens=2, temperature=args.temperature)
    prefill_result = bench_generate(engine,prompt_tokens, 1,2,args.temperature)
    prefill_time = prefill_result["ttft"]
    prefill_mfu = 100 * model.estimate_prefill_flops(prompt_len) / prefill_time / peak_flops
    prefill_tok_per_sec = prompt_len / prefill_time

    print(f"Prefill (batch 1, {prompt_len} tokens): {prefill_tok_per_sec:,.0f} tok/s | MFU {prefill_mfu:.1f}%")
    payload["prefill"] = {
        "tok_per_sec" : round(prefill_tok_per_sec,1),
        "mfu_percent" : round(prefill_mfu,2),
        "time_sec" : round(prefill_time,6),
    }


    # batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    header = f"{'batch':>6} {'TTFT ms' : >9} {'TPOT ms' : >9} {'tok/s':>10} {'MBU %' : >7} {'MFU %':>7} {'VRAM GiB' :>9} {'steps' : >6}"

    print(header)

    print("-" * len(header))

    for batch_size in batch_sizes:
        bench_generate(engine, prompt_tokens, batch_size, 8,args.temperature)

        result = bench_generate(engine, prompt_tokens, batch_size, args.decode_tokens, args.temperature)
        step_times = result["step_times"]
        num_steps = len(step_times)
        if num_steps ==0:
            print(f"{batch_size:>6} all rows terminated during warmup?! Skipping!!!")
            continue

        tpot= sorted(step_times)[num_steps//2]
        tok_per_sec = batch_size * num_steps / sum(step_times)

        bytes_per_step = w_bytes + batch_size * kv_read


        mbu = 100 * (bytes_per_step/tpot) / peak_bw

        flops_per_step = batch_size * model.estimate_decode_flops(context_mid)

        mfu = 100 * (flops_per_step /tpot) / peak_flops
        vram_gib = result["peak_vram"]/2**30

        note = "" if num_steps == args.decode_tokens -1 else f" (Early Stop @ {num_steps})"

        print (f"{batch_size:>6} {result['ttft']*1e3:>9.1f} {tpot*1e3:>9.2f} {tok_per_sec:>10,.0f} "
              f"{mbu:>7.1f} {mfu:>7.2f} {vram_gib:>9.2f} {num_steps:>6}{note}")

        payload["sweep"].append({
            "batch_size" : batch_size,
            "ttft_sec" : round(result["ttft"], 6),
            "tpot_sec" : round(tpot, 6),
            "tok_per_sec": round(tok_per_sec, 1),
            "mbu_percent" : round(mbu,2),
            "mfu_percent": round(mfu,4),

            "peak_vram_bytes" : result["peak_vram"],
            "decode_steps" : num_steps,

        })


    print("-" * len(header))
    print(json.dumps(payload))

    compute_cleanup()

if __name__=="__main__":
    main()
    
