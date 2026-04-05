import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
from contextlib import nullcontext

import torch

from nanollm.commons import compute_init,compute_cleanup, print0,get_base_dir, autodetect_device_type, download_file_with_lock
from nanollm.tokenizer import get_token_bytes
from nanollm.checkpoint_manager import load_model
from nanollm.core_eval import evaluate_task
from nanollm.dataloader import tokenizing_distributed_data_loader_with_bos_bestfit
from nanollm.loss_eval import evaluate_bpb
from nanollm.engine import Engine


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    base_dir=get_base_dir()
    eval_bundle_dir=os.path.join(base_dir,"eval_bundle")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path,'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir=os.path.join(tmpdir,"eval_bundle")
        shutil.move(extracted_bundle_dir,eval_bundle_dir)

    print0(f"Eval bundle placed at {eval_bundle_dir}")


def evaluate_core(model,tokenizer,device,max_per_task=1):
    base_dir=get_base_dir()
    eval_bundle_dir=os.path.join(base_dir,"eval_bundle")

    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)


    config_path=os.path.join(eval_bundle_dir,'core.yaml')
    data_base_path=os.path.join(eval_bundle_dir,"eval_data")
    eval_meta_data=os.path.join(eval_bundle_dir,"eval_meta_data.csv")

    with open(config_path,'r',encoding='utf-8') as f:
        config=yaml.safe_load(f)
    tasks=config['icl_tasks']

    random_baselines={}

    with open(eval_meta_data,'r',encoding='utf-8') as f:
        reader=csv.DictReader(f)

        for row in reader:
            task_name=row["Eval Task"]
            random_baseline=row["Random baseline"]
            random_baselines[task_name]= float(random_baseline)

    results={}
    centered_results={}

    for task in tasks:
        start_time=time.time()
        label=task['label']
        task_meta={
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot' : task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})...", end='')

        data_path=os.path.join(data_base_path,task_meta['dataset_uri'])

        with open(data_path, 'r',encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng=random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task>0:
            data=data[:max_per_task]


        accuracy=evaluate_task(model,tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result=(accuracy-0.01 * random_baseline) /(1.0-0.01*random_baseline)
        centered_results[label] =centered_result
        elapsed=time.time() - start_time
        print0(f"accuracy: {accuracy: .4f} | centered: {centered_result: .4f} | time: {elapsed:.2f}s")


    
    core_metric=sum(centered_results.values()) /len(centered_results)

    out ={
        "results":results,
        "centered_results":centered_results,
        "core_metric": core_metric
    }

    return out


def main():
    parser=argparse.ArgumentParser(description="Base model evalutation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanollm model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()


    eval_modes=set(mode.strip() for mode in args.eval.split(','))
    valid_cores={'core','bpb','sample'}
    invalid=eval_modes - valid_cores
    if invalid:
        parser.error(f"Invalid eval modes specified: {invalid}. Valid options are: {valid_cores}")
        

    device_type=autodetect_device_type() if args.device_type =='' else args.device_type
    ddp,ddp_rank,ddp_local_rank, ddp_world_size,device=compute_init(device_type)
    # autocast_ctx=torch.autocast(device_type=device_type,dtype=torch.bfloat16) if device_type=='cuda' else nullcontext()
    
    model,tokenizer,meta=load_model("base",device, phase="eval",model_tag=args.model_tag,step=args.step)

    sequence_len=meta["model_config"]["sequence_len"]
    token_bytes=get_token_bytes(device=device)
    model_name=f"base_model (step {meta['step']})"
    model_slug=f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval Nodes: {','.join(sorted(eval_modes))}")

    core_results=None
    bpb_results={}
    samples=[]
    unconditional_samples=[]

    if 'sample' in eval_modes:
        print0("\n"+"="*80)
        print0("Model Samples")
        print0("\n"+"="*80)

        if ddp_rank==0:
            prompts=[
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            

            engine=Engine(model,tokenizer)
            print0("\nConditioned Samples:")

            for prompt in prompts:
                tokens=tokenizer(prompt, prepend="<|bos|>")

                sample, _ =engine.generate_batch(tokens,num_samples=1, max_tokens=16,temperature=0)

                sample_str=tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)


            print0("\nUnconditioned samples:")

            tokens=tokenizer("", prepend="<|bos|>")

            uncond,_ = engine.generate_batch(tokens,num_samples=8,max_tokens=128,temperature=1.0)

            for sample in uncond:
                sample_str=tokenizer.decode(sample)
                print0("-"*80)
                print0(sample_str)
                unconditional_samples.append(sample_str)

    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)

        tokens_per_step= args.device_batch_size * sequence_len * ddp_world_size

        if args.split_tokens % tokens_per_step != 0:
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step

            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")

        steps= args.split_tokens //tokens_per_step

        for split_name in ["train", "val"]:
            loader= tokenizing_distributed_data_loader_with_bos_bestfit(
                tokenizer, args.device_batch_size,
                sequence_len, split_name,
                device=device
            )

            bpb = evaluate_bpb(model, loader, steps, token_bytes)

            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb : {bpb:.6f}")


    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)

        core_results = evaluate_core( model, tokenizer, device, max_per_task=args.max_per_task)

        
        if ddp_rank==0:
            base_dir=get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

            with open(output_csv_path, 'w', envoding= 'utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered': <10}\n")
                for label in core_results["results"]:
                    acc = core_results['results'][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")

                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")

            print0(f"\n Results written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    
    from nanollm.report import get_report

    report_data =[ {"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")


    if samples:
        report_data.append({f"Sample {i}": s for i,s in enumerate(samples)})


    if unconditional_samples:
        report_data.append({f"Unconditiioned {i}": s for i,s in enumerate(unconditional_samples)})

    get_report().log(section = "Base model evaluation", data= report_data)

    compute_cleanup()


if __name__=="__main__":
    main()