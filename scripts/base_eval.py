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

from nanollm.common import compute_init,compute_cleanup, print0,get_base_dir, autodetect_device_type, download_file_with_lock
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
            'new_fewshot' : task['new_fewshot'][0],
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
        random_baseline = random_baseline[label]
        centered_result=(accuracy-0.01 * random_baseline) /(1.0-0.01*random_baseline)
        centered_results[label] =centered_result
        elapsed=time.time() - start_time
        print0(f"accuracy: {accuracy: .4f} | centered: {centered_results: .4f} | time: {elapsed:.2f}s")

    
    core_metric=sum(centered_results.values()) /len(centered_results)

    out ={
        "results":results,
        "centered_results":centered_results,
        "core_metric": core_metric
    }

    return out