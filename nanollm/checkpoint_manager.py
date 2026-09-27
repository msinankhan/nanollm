import os
import re
import glob
import json
import torch
import logging
import hashlib
import shutil
import tempfile
import threading
import uuid

from nanollm.tokenizer import get_tokenizer
from nanollm.commons import get_base_dir
from nanollm.gpt import GPT, GPTConfig
from nanollm.commons import setup_default_logging

setup_default_logging()
logger=logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1
TRAINING_COMPLETE_FILENAME = "training.complete.json"

def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def _checkpoint_filenames(step, rank=0, has_optimizer=True):
    names = [f"model_{step:06d}.pt", f"model_{step:06d}.json"]
    if has_optimizer:
        names.append(f"optim_{step:06d}_rank{rank:d}.pt")
    return names

def _complete_path(checkpoint_dir, step):
    return os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.complete.json")

def _training_complete_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, TRAINING_COMPLETE_FILENAME)

def mark_training_complete(checkpoint_dir, step, num_iterations, metadata=None):
    """Publish a small marker only after the final checkpoint is durable."""
    _validate_committed_checkpoint(checkpoint_dir, step, require_optimizer=False)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "num_iterations": num_iterations,
        "complete": True,
    }
    if metadata:
        payload.update(metadata)
    _atomic_json_save(payload, _training_complete_path(checkpoint_dir))
    logger.info(f"Marked training complete at step {step} in {checkpoint_dir}")

def clear_training_complete(checkpoint_dir):
    path = _training_complete_path(checkpoint_dir)
    if os.path.exists(path):
        os.remove(path)

def read_training_complete(checkpoint_dir):
    path = _training_complete_path(checkpoint_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("complete") is not True:
        raise ValueError(f"Invalid training completion marker: {path}")
    _validate_committed_checkpoint(checkpoint_dir, int(payload["step"]))
    return payload

def log0(message):
    if int(os.environ.get('RANK',0))==0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    if "window_pattern" not in  model_config_kwargs:
        model_config_kwargs["window_pattern"] =  "L"
        log0(f"Patching missing window_pattern config to {model_config_kwargs["window_pattern"]}")


def _patch_missing_keys(model_data,model_config):
    n_layer=model_config.n_layer

    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1 .")

    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0 .")

def _atomic_json_save(data, path):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def _write_checkpoint_bundle(staging_root, step, model_data, optimizer_data, meta_data, rank=0):
    os.makedirs(staging_root, exist_ok=True)
    bundle_dir = tempfile.mkdtemp(prefix=f"checkpoint_{step:06d}_", dir=staging_root)
    try:
        model_name, meta_name = _checkpoint_filenames(step, rank, False)
        model_path = os.path.join(bundle_dir, model_name)
        meta_path = os.path.join(bundle_dir, meta_name)
        torch.save(model_data, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if optimizer_data is not None:
            optimizer_name = f"optim_{step:06d}_rank{rank:d}.pt"
            torch.save(optimizer_data, os.path.join(bundle_dir, optimizer_name))
        filenames = _checkpoint_filenames(step, rank, optimizer_data is not None)
        files = {}
        for name in filenames:
            path = os.path.join(bundle_dir, name)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                raise IOError(f"Checkpoint file was not written correctly: {path}")
            files[name] = {"size": os.path.getsize(path), "sha256": _sha256(path)}
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": step,
            "rank": rank,
            "has_optimizer": optimizer_data is not None,
            "files": files,
        }
        with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        return bundle_dir, manifest
    except Exception:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise

def _publish_checkpoint_bundle(bundle_dir, checkpoint_dir, manifest, keep_last=2):
    os.makedirs(checkpoint_dir, exist_ok=True)
    step = manifest["step"]
    upload_id = uuid.uuid4().hex
    uploaded = []
    try:
        for name, expected in manifest["files"].items():
            source = os.path.join(bundle_dir, name)
            temporary = os.path.join(checkpoint_dir, f".{name}.uploading-{upload_id}")
            shutil.copyfile(source, temporary)
            if os.path.getsize(temporary) != expected["size"] or _sha256(temporary) != expected["sha256"]:
                raise IOError(f"Persistent checkpoint verification failed: {temporary}")
            uploaded.append((temporary, os.path.join(checkpoint_dir, name)))
        marker = _complete_path(checkpoint_dir, step)
        if os.path.exists(marker):
            os.remove(marker)
        for temporary, final in uploaded:
            os.replace(temporary, final)
        _atomic_json_save(manifest, marker)
        logger.info(f"Committed complete checkpoint step {step} to {checkpoint_dir}")
        if keep_last > 0:
            _prune_checkpoints(checkpoint_dir, keep_last)
    finally:
        for temporary, _ in uploaded:
            if os.path.exists(temporary):
                os.remove(temporary)
        shutil.rmtree(bundle_dir, ignore_errors=True)

def _prune_checkpoints(checkpoint_dir, keep_last):
    markers = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.complete.json")))
    for marker in markers[:-keep_last]:
        try:
            with open(marker, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for name in manifest.get("files", {}):
                path = os.path.join(checkpoint_dir, name)
                if os.path.exists(path):
                    os.remove(path)
            os.remove(marker)
        except Exception as exc:
            logger.warning(f"Could not prune checkpoint marker {marker}: {exc}")

class AsyncCheckpointWriter:
    """Stage a consistent snapshot locally, then publish it in the background."""
    def __init__(self, checkpoint_dir, staging_dir=None, keep_last=2):
        self.checkpoint_dir = checkpoint_dir
        self.staging_dir = staging_dir or os.path.join(tempfile.gettempdir(), "nanollm-checkpoints")
        self.keep_last = keep_last
        self._thread = None
        self._error = None

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("Background checkpoint publish failed") from error

    def submit(self, step, model_data, optimizer_data, meta_data, rank=0):
        self.wait()
        bundle_dir, manifest = _write_checkpoint_bundle(
            self.staging_dir, step, model_data, optimizer_data, meta_data, rank
        )
        logger.info(f"Validated local checkpoint step {step}; publishing in background")
        def publish():
            try:
                _publish_checkpoint_bundle(
                    bundle_dir, self.checkpoint_dir, manifest, self.keep_last
                )
            except Exception as exc:
                self._error = exc
                logger.exception("Background checkpoint publish failed")
        self._thread = threading.Thread(target=publish, name=f"checkpoint-{step}", daemon=False)
        self._thread.start()

def save_checkpoint(checkpoint_dir,step,model_data, optimizer_data, meta_data,rank=0):
    staging = os.path.join(checkpoint_dir, ".staging")
    bundle_dir, manifest = _write_checkpoint_bundle(
        staging, step, model_data, optimizer_data, meta_data, rank
    )
    _publish_checkpoint_bundle(bundle_dir, checkpoint_dir, manifest, keep_last=0)

def _validate_committed_checkpoint(checkpoint_dir, step, require_optimizer=False, rank=0):
    marker = _complete_path(checkpoint_dir, step)
    if not os.path.isfile(marker):
        raise FileNotFoundError(f"Checkpoint step {step} has no completion marker: {marker}")
    with open(marker, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("step") != step:
        raise ValueError(f"Checkpoint marker step mismatch in {marker}")
    if require_optimizer and not manifest.get("has_optimizer"):
        raise FileNotFoundError(f"Checkpoint step {step} has no optimizer state")
    for name, expected in manifest.get("files", {}).items():
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) != expected["size"]:
            raise IOError(f"Checkpoint file missing or has wrong size: {path}")
        if _sha256(path) != expected["sha256"]:
            raise IOError(f"Checkpoint checksum mismatch: {path}")
    return manifest

def load_checkpoint(checkpoint_dir,step,device, load_optimizer=False, rank=0):
    _validate_committed_checkpoint(checkpoint_dir, step, load_optimizer, rank)
    model_path=os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data=torch.load(model_path, map_location=device)

    optimizer_data=None
    if load_optimizer:
        optimizer_path=os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data=torch.load(optimizer_path, map_location=device)

    meta_path=os.path.join(checkpoint_dir, f"model_{step:06d}.json")
    with open(meta_path, 'r',encoding="utf-8") as f:
        meta_data=json.load(f)
    return model_data,optimizer_data,meta_data



def build_model(checkpoint_dir,step, device,phase):
    assert phase in ["train","eval"], f"The phase must be either train or eval:{phase}."

    model_data,optimizer_data,meta_data=load_checkpoint(checkpoint_dir,step,device, load_optimizer=False)

    if device.type in {"cpu"}:
        model_data={k:v.float() if v.dtype==torch.bfloat16 else v 
                     for k,v in model_data.items()}
        

    model_data={k.removeprefix("_orig_mod."): v for k,v in model_data.items()}
    model_config_kwargs=meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config=GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)

    with torch.device("meta"):
        model=GPT(model_config)


    model.to_empty(device=device)
    model.init_weights() #Some model components (e.g. rotary embeddings) are buffers and are not part of state_dict. They need initialization logic to run.
                        #Hence we have to run init_weights() in spite of most parameters getting overwritten.
    model.load_state_dict(model_data,strict=True,assign=True)

    if phase=="eval":
        model.eval()         #disables dropout, freezes layernorm behavior
    else:
        model.train()        #enables stochastic layers.

    tokenizer=get_tokenizer()

    assert tokenizer.get_vocab_size() ==model_config_kwargs["vocab_size"], f"Tokenizer Vocab Size {tokenizer.get_vocab_size()} doesn't match config vocab size {model_config_kwargs["vocab_size"]}"
    return model,tokenizer,meta_data
        
def find_largest_model(checkpoint_dir):
    model_tags=[f for f in os.listdir(checkpoint_dir) if  os.path.isdir(os.path.join(checkpoint_dir,f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    candidates=[]
    for model_tag in model_tags:
        match=re.match(r"d(\d+)",model_tag)

        if match:
            model_depth=int(match.group(1))
            candidates.append((model_depth,model_tag))

    if candidates:
        candidates.sort(key=lambda x:x[0], reverse=True)
        return candidates[0][1]
    
    model_tags.sort(key=lambda x:os.path.getmtime(os.path.join(checkpoint_dir,x)), reverse=True)
    return model_tags[0]

def find_last_step(checkpoint_dir):
    markers = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.complete.json"))
    if not markers:
        raise FileNotFoundError(f"No complete checkpoints found in {checkpoint_dir}")
    steps = sorted(
        (int(os.path.basename(path).split("_")[1].split(".")[0]) for path in markers),
        reverse=True,
    )
    errors = []
    for step in steps:
        try:
            _validate_committed_checkpoint(checkpoint_dir, step)
            return step
        except Exception as exc:
            errors.append(f"step {step}: {exc}")
    raise IOError("No valid completed checkpoint: " + " | ".join(errors))

def load_model_from_dir(checkpoint_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        model_tag=find_largest_model(checkpoint_dir)
        log0(f"No model tag provided, guessing the model tag:{model_tag}")

    checkpoint_dir=os.path.join(checkpoint_dir,model_tag)

    if step is None:
        step=find_last_step(checkpoint_dir)

    assert step is not None, f"No checkpoints found in {checkpoint_dir}"

    log0(f"Loading model from {checkpoint_dir} with step {step}")

    model,tokenizer, meta_data=build_model(checkpoint_dir,step,device,phase)
    return model,tokenizer,meta_data


def load_model(source,*args,**kwargs):
    model_dir={
        "base":"base_checkpoints",
        "mid":"mid_checkpoints",
        "sft":"chatsft_checkpoints",
        "rl": "chatrl_checkpoints"
    }[source]

    base_dir=get_base_dir()
    checkpoint_dir=os.path.join(base_dir,model_dir)
    return load_model_from_dir(checkpoint_dir,*args,**kwargs)

def load_optimizer_state(source, device,rank, model_tag=None, step=None):
    model_dir={
        "base" : "base_checkpoints",
        "sft" : "chatsft_checkpoints",
        "rl" : "chatrl_checkpoints"
    }[source]

    base_dir = get_base_dir()

    checkpoints_dir= os.path.join(base_dir,model_dir)

    if model_tag is None:
        model_tag=find_largest_model(checkpoints_dir)

    checkpoint_dir=os.path.join(checkpoints_dir,model_tag)

    if step is None:
        step= find_last_step(checkpoint_dir)

    _validate_committed_checkpoint(checkpoint_dir, step, require_optimizer=True, rank=rank)
    optimizer_path= os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data=torch.load(optimizer_path,map_location=device)
    return optimizer_data
