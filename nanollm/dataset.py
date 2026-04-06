import os
import time
import argparse
import requests 
import json
import subprocess
import shutil
from pathlib import Path
from multiprocessing import Pool

# Use huggingface_hub for robust downloads
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Warning: huggingface_hub not installed. Installing...")
    subprocess.run(["pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import hf_hub_download

from nanollm.commons import get_base_dir


# ============================================================
# Parameter Golf Challenge Dataset Setup
# ============================================================
# This downloads the FineWeb dataset in the exact format used
# by the parameter-golf challenge (1024 BPE tokenizer, bin files)

# Default: Use the published parameter-golf data export
DEFAULT_DATA_REPO = "willdepueoai/parameter-golf"
DEFAULT_VARIANT = "sp1024"  # 1024 BPE tokenizer (the default/canonical one)

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "datasets", "fineweb10B_sp1024")
TOKENIZER_DIR = os.path.join(base_dir, "data", "tokenizers")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)


def get_hf_token():
    """Get HuggingFace token from environment if set."""
    return os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", None))


def download_file_hf(relative_path: str, local_path: str, repo_id: str = DEFAULT_DATA_REPO):
    """
    Download a single file using huggingface_hub.
    
    Args:
        relative_path: Path within the HF repo (e.g., 'tokenizers/sp1024.model')
        local_path: Where to save the file locally
        repo_id: HuggingFace repo ID
    """
    if os.path.exists(local_path):
        return local_path
    
    # Determine subfolder from the path
    remote_path = Path(relative_path)
    subfolder = remote_path.parent.as_posix() if remote_path.parent != Path(".") else None
    filename = remote_path.name
    
    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            repo_type="dataset",
            token=get_hf_token(),
        )
        
        # Copy from HF cache to our desired location
        # This ensures we have a real file, not a symlink
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(cached_path, local_path)
        
        return local_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to download {relative_path}: {e}")


def dataset_dir_for_variant(variant: str) -> str:
    """Get the dataset directory name for a variant."""
    if variant.startswith("sp") and variant[2:].isdigit():
        return f"fineweb10B_{variant}"
    raise ValueError(f"Unsupported variant: {variant}")


def download_parameter_golf_data(variant: str = DEFAULT_VARIANT, train_shards: int = 80, val_shards: int = 1):
    """
    Download the FineWeb dataset in parameter-golf format.
    
    This uses the cached export from HuggingFace that matches exactly what
    the challenge uses (same tokenizer, same data split, same tokenization).
    
    Args:
        variant: Tokenizer variant (default: sp1024 = 1024 BPE)
        train_shards: Number of training shards to download (default: 80 = ~8B tokens)
        val_shards: Number of validation shards (default: 1)
    """
    print(f"[Parameter Golf] Downloading FineWeb dataset (variant: {variant})")
    print(f"[Parameter Golf] Target directory: {DATA_DIR}")
    print(f"[Parameter Golf] Using repo: {DEFAULT_DATA_REPO}")
    
    # Dataset subfolder in HF repo
    dataset_subfolder = dataset_dir_for_variant(variant)
    
    # First, download the tokenizer
    # HF names it like: fineweb_1024_bpe.model (vocab size from variant)
    vocab_size = variant[2:] if variant.startswith("sp") else variant
    tokenizer_filename = f"fineweb_{vocab_size}_bpe.model"
    tokenizer_path = os.path.join(TOKENIZER_DIR, tokenizer_filename)
    
    if not os.path.exists(tokenizer_path):
        print(f"[Parameter Golf] Downloading tokenizer...")
        try:
            download_file_hf(
                relative_path=f"tokenizers/{tokenizer_filename}",
                local_path=tokenizer_path,
            )
            print(f"[Parameter Golf] Tokenizer saved to: {tokenizer_path}")
        except Exception as e:
            print(f"[Parameter Golf] Error downloading tokenizer: {e}")
            raise
    else:
        print(f"[Parameter Golf] Tokenizer already exists: {tokenizer_path}")
    
    # Download training data
    print(f"[Parameter Golf] Downloading {train_shards} training shards...")
    for shard_idx in range(train_shards):
        shard_filename = f"fineweb_train_{shard_idx:05d}.bin"
        shard_file = os.path.join(DATA_DIR, shard_filename)
        
        if not os.path.exists(shard_file):
            try:
                download_file_hf(
                    relative_path=f"{dataset_subfolder}/{shard_filename}",
                    local_path=shard_file,
                )
                print(f"  Downloaded: {shard_filename}")
            except Exception as e:
                print(f"  Warning: Failed to download {shard_filename}: {e}")
        else:
            print(f"  Skipping {shard_filename} (already exists)")
    
    # Download validation data
    print(f"[Parameter Golf] Downloading {val_shards} validation shard(s)...")
    for shard_idx in range(val_shards):
        shard_filename = f"fineweb_val_{shard_idx:05d}.bin"
        shard_file = os.path.join(DATA_DIR, shard_filename)
        
        if not os.path.exists(shard_file):
            try:
                download_file_hf(
                    relative_path=f"{dataset_subfolder}/{shard_filename}",
                    local_path=shard_file,
                )
                print(f"  Downloaded: {shard_filename}")
            except Exception as e:
                print(f"  Warning: Failed to download {shard_filename}: {e}")
        else:
            print(f"  Skipping {shard_filename} (already exists)")
    
    print(f"\n[Parameter Golf] Dataset download complete!")
    print(f"[Parameter Golf] Data directory: {DATA_DIR}")
    print(f"[Parameter Golf] Tokenizer: {tokenizer_path}")
    
    return DATA_DIR, tokenizer_path


# ============================================================
# Data Loading Functions for Parameter Golf Format
# ============================================================

import numpy as np
import torch


def list_bin_files(split="train", data_dir=None):
    """List all binary data files for train or val split."""
    data_dir = data_dir or DATA_DIR
    pattern = f"fineweb_{split}_*.bin"
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin') and pattern.replace('*', '') in f])
    return [os.path.join(data_dir, f) for f in files]


def load_bin_shard(filepath):
    """Load a single binary shard file.
    
    The files contain token IDs as uint16 (2 bytes per token).
    Each file is a sequence of tokens that can be any length.
    """
    data = np.fromfile(filepath, dtype=np.uint16)
    return data


def get_tokenizer_path(variant="sp1024"):
    """Get path to the tokenizer for the given variant.
    
    The tokenizer files on HF are named like: fineweb_1024_bpe.model
    (where 1024 is extracted from the variant 'sp1024')
    """
    # Extract vocab size from variant (e.g., "sp1024" -> "1024")
    vocab_size = variant[2:] if variant.startswith("sp") else variant
    return os.path.join(TOKENIZER_DIR, f"fineweb_{vocab_size}_bpe.model")


def decode_bin_to_text(
    data_dir: str = None,
    variant: str = "sp1024",
    output_dir: str = None,
):
    """Decode the binary token files back to text.
    
    This allows you to get the raw text from the parameter-golf dataset
    so you can re-tokenize it with your own tokenizer (e.g., rustbpe).
    
    Args:
        data_dir: Directory containing binary shards (default: DATA_DIR)
        variant: Tokenizer variant
        output_dir: Where to save the decoded text (default: <base>/data/text)
    """
    try:
        import sentencepiece as spm
    except ImportError:
        print("Installing sentencepiece...")
        subprocess.run(["pip", "install", "sentencepiece", "-q"], check=True)
        import sentencepiece as spm
    
    data_dir = data_dir or DATA_DIR
    output_dir = output_dir or os.path.join(base_dir, "data", "text", f"fineweb10B_{variant}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = get_tokenizer_path(variant)
    print(f"[Decode] Loading tokenizer: {tokenizer_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    print(f"[Decode] Vocab size: {sp.get_piece_size()}")
    
    # Find all binary files
    train_files = sorted([f for f in os.listdir(data_dir) if f.startswith("fineweb_train_") and f.endswith(".bin")])
    val_files = sorted([f for f in os.listdir(data_dir) if f.startswith("fineweb_val_") and f.endswith(".bin")])
    
    print(f"[Decode] Found {len(train_files)} train shards, {len(val_files)} val shards")
    
    def decode_shard(filepath, output_path):
        """Decode a single binary shard to text."""
        tokens = np.fromfile(filepath, dtype=np.uint16)
        text = sp.decode(tokens.tolist())
        
        # Save as JSONL (one JSON object per line with "text" field)
        with open(output_path, 'w', encoding='utf-8') as f:
            # Split by common delimiters to create multiple entries
            # This preserves the structure roughly
            for line in text.split('\n'):
                if line.strip():
                    json.dump({"text": line}, f, ensure_ascii=False)
                    f.write('\n')
        
        return len(tokens)
    
    # Decode training files
    print("[Decode] Decoding training data...")
    for i, fname in enumerate(train_files):
        bin_path = os.path.join(data_dir, fname)
        text_path = os.path.join(output_dir, fname.replace(".bin", ".jsonl"))
        
        if os.path.exists(text_path):
            print(f"  Skipping {fname} (already exists)")
            continue
        
        num_tokens = decode_shard(bin_path, text_path)
        print(f"  Decoded {fname} -> {num_tokens} tokens")
    
    # Decode validation files
    print("[Decode] Decoding validation data...")
    for i, fname in enumerate(val_files):
        bin_path = os.path.join(data_dir, fname)
        text_path = os.path.join(output_dir, fname.replace(".bin", ".jsonl"))
        
        if os.path.exists(text_path):
            print(f"  Skipping {fname} (already exists)")
            continue
        
        num_tokens = decode_shard(bin_path, text_path)
        print(f"  Decoded {fname} -> {num_tokens} tokens")
    
    print(f"[Decode] Complete! Text saved to: {output_dir}")


# ============================================================
# Legacy FineWeb-Edu Dataset (karpathy's version)
# ============================================================

BASE_URL="https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARDS=1822

index_to_filename=lambda index:f"shard_{index:05d}.parquet"


def list_parquet_files(data_dir=None):
    data_dir=DATA_DIR if data_dir is None else data_dir

    parquet_files=sorted([
        f for f in os.listdir(data_dir)if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    parquet_paths=[os.path.join(data_dir,f) for f in parquet_files]

    return parquet_paths


def parquet_iter_batches(split,start=0,step=1):

    assert split in ["train" , "val"], "Invalid split, must be 'train' or 'val'."
    parquet_paths=list_parquet_files()
    parquet_paths=parquet_paths[:-1] if split=='train' else parquet_paths[-1:]

    for file in parquet_paths:
        pf=pq.ParquetFile(file)
        for rg_idx in range (start,pf.num_row_groups,step):
            rg=pf.read_row_group(rg_idx)
            texts=rg.column('text').to_pylist()
            yield texts



def download_single_file(index):
    filename=index_to_filename(index)
    file_path=os.path.join(DATA_DIR,filename)

    if os.path.exists(file_path):
        print(f"Skipping {file_path} as it already exists.")
        return True
    
    url=f"{BASE_URL}/{filename}"
    print(f"Downloading file {filename}")

    max_attempts=5

    for attempt in range(1,max_attempts+1):
        try:
            response=requests.get(url=url,stream=True,timeout=30)
            response.raise_for_status()


            temp_path=file_path +f".tmp"

            with open(temp_path,'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)

            os.rename(temp_path,file_path)

            print(f"Sucessfully Downloaded {filename}")

            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename} : {e}")

            for path in [file_path +f".tmp",file_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass


            if attempt<max_attempts:
                wait_time=2**attempt
                print(f"Wating {wait_time} seconds before retrying Download.")
                time.sleep(wait_time)

            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Download dataset for pretraining.")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="download",
                        choices=["download", "decode"],
                        help="Mode: download (binary files) or decode (binary -> text)")
    
    # Dataset source options
    parser.add_argument("--source", type=str, default="parameter-golf", 
                        choices=["parameter-golf", "karpathy"],
                        help="Dataset source: parameter-golf (challenge format) or karpathy (fineweb-edu)")
    
    # Parameter golf specific options
    parser.add_argument("--variant", type=str, default="sp1024",
                        help="Tokenizer variant for parameter-golf (default: sp1024)")
    parser.add_argument("--train-shards", type=int, default=80,
                        help="Number of training shards for parameter-golf (default: 80)")
    parser.add_argument("--val-shards", type=int, default=1,
                        help="Number of validation shards for parameter-golf (default: 1)")
    
    # Karpathy specific options
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (karpathy), -1=all")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of Parallel download workers (default:4)")

    args=parser.parse_args()

    if args.mode == "decode":
        # Decode binary to text
        print("=" * 60)
        print("Decoding Binary Tokens to Text")
        print("=" * 60)
        decode_bin_to_text(
            variant=args.variant,
        )
    elif args.source == "parameter-golf":
        print("=" * 60)
        print("Downloading Parameter Golf Challenge Dataset")
        print("=" * 60)
        download_parameter_golf_data(
            variant=args.variant,
            train_shards=args.train_shards,
            val_shards=args.val_shards
        )
    else:
        # Legacy karpathy dataset
        num=MAX_SHARDS+1 if args.num_files==-1 else min(args.num_files, MAX_SHARDS+1)
        ids_to_download=list(range(num))

        print(f"Downlading {len(ids_to_download)} shards using {args.num_workers} workers...")
        print(f"Target Directory{DATA_DIR}")
        print()

        # Update DATA_DIR for karpathy source
        DATA_DIR_KARPATHY = os.path.join(base_dir, "base_data")
        os.makedirs(DATA_DIR_KARPATHY, exist_ok=True)

        with Pool(processes=args.num_workers) as pool:
            results= pool.map(download_single_file, ids_to_download)


        successful= sum(1 for success in results if success)

        print(f"Done! Downloaded:{successful}/{len(ids_to_download)} shards to {DATA_DIR_KARPATHY}")