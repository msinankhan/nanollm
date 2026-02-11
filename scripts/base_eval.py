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

from nanollm.common import compute_init,compute_cleanup, print0,get_base_dir, autodetect_device_type, download_file_with lock
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


# def evaluate_core(model,tokenizer,device,max_per_task=1):
