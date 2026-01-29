import os
import re
import glob
import json
import torch
import logging

from nanollm.tokenizer import get_tokenizer
from nanollm.commons import get_base_dir
from nanollm.gpt import GPT, GPTConfig
from nanollm.commons import setup_default_logging

setup_default_logging()
logger=logging.getLogger(__name__)

def log0(message):
    if int(os.environ.get('RANK',0))==0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    if "window_pattern" not in  model_config_kwargs:
        model_config_kwargs["window_pattern"] =  "L"
        log0(f"Patching missing window_pattern config to {model_config_kwargs["window_pattern"]}")


def _patch_missing_keys(model_data,model_config):
    n_layer=model_config.n_layer

    if "resid_lambda" not in model_data:
        model_data["resid_lambda"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambda in model data to 1 .")

    if "x0_lambdas" not in model_data:
        model_data["x0_lambda"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambda in model data to 0.0 .")


        
