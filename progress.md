# Parameter Golf Dataset Integration - Progress Notes

## Overview

This document tracks the integration of the parameter-golf dataset into nanollm, enabling training with either the pre-tokenized binary format or with a custom tokenizer (rustbpe).

---

## 1. Parameter Golf Repository

**Repository**: [openai/parameter-golf](https://github.com/openai/parameter-golf) (via HuggingFace)

**HuggingFace Dataset**: [willdepueoai/parameter-golf](https://huggingface.co/datasets/willdepueoai/parameter-golf)

### Repository Structure

```
willdepueoai/parameter-golf/
├── datasets/
│   ├── tokenizers/
│   │   ├── fineweb_1024_bpe.model    # SentencePiece tokenizer
│   │   └── fineweb_1024_bpe.vocab
│   ├── datasets/                      # Binary token files
│   │   └── fineweb10B_sp1024/
│   │       ├── fineweb_train_*.bin   # Training shards (80 files)
│   │       └── fineweb_val_*.bin     # Validation shard
│   ├── docs_selected.jsonl           # Raw documents (for training custom tokenizers)
│   └── manifest.json
├── data/
│   ├── cached_challenge_fineweb.py   # Main download script
│   ├── download_hf_docs_and_tokenize.py  # Train custom tokenizer
│   └── tokenizer_specs.json
```

### Key Details

- **Tokenizer**: SentencePiece BPE with vocab_size=1024
- **Dataset**: FineWeb 10B tokens (80 training shards + 1 validation)
- **Format**: Binary uint16 tokens (.bin files)
- **Variant**: `sp1024` (maps to `fineweb10B_sp1024` folder)

---

## 2. Changes to nanollm/dataset.py

### 2.1 Changed Download Method: curl → huggingface_hub

**Before**: Used subprocess with curl to download files
**After**: Uses `huggingface_hub.hf_hub_download()`

Benefits:
- Proper HuggingFace cache handling
- Symlink resolution (avoids broken symlinks)
- Automatic token handling via `HF_TOKEN` env var
- More robust error handling

Key changes:
```python
# Added import with auto-install
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.run(["pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import hf_hub_download

# New download function
def download_file_hf(relative_path, local_path, repo_id):
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        repo_type="dataset",
        token=get_hf_token(),
    )
    shutil.copy2(cached_path, local_path)
```

### 2.2 Fixed Tokenizer File Naming

**Issue**: HF uses `fineweb_1024_bpe.model`, but code was saving as `sp1024.model`

**Fix**: Extract vocab_size from variant and construct correct filename:
```python
def get_tokenizer_path(variant="sp1024"):
    vocab_size = variant[2:] if variant.startswith("sp") else variant
    return os.path.join(TOKENIZER_DIR, f"fineweb_{vocab_size}_bpe.model")
```

### 2.3 Added Decode Functionality

Added `decode_bin_to_text()` function to decode binary tokens back to text, enabling training with custom tokenizers (like rustbpe).

```python
def decode_bin_to_text(
    data_dir=None,
    variant="sp1024",
    output_dir=None,
):
    """Decode binary token files back to text (JSONL format)."""
    # Load SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(get_tokenizer_path(variant))
    
    # Decode each binary shard to text
    for each train/val .bin file:
        tokens = np.fromfile(filepath, dtype=np.uint16)
        text = sp.decode(tokens.tolist())
        # Save as JSONL
```

---

## 3. Two Modes of Operation

### Mode 1: Download Binary (Pre-tokenized)

Downloads the pre-tokenized binary files for training with the original tokenizer.

```bash
python -m nanollm.dataset
python -m nanollm.dataset --mode download
python -m nanollm.dataset --train-shards 5 --val-shards 1  # For testing
```

**Output**:
```
<nanollm_dir>/data/
├── datasets/fineweb10B_sp1024/
│   ├── fineweb_train_00000.bin
│   ├── fineweb_train_00001.bin
│   └── ...
│   └── fineweb_val_00000.bin
└── tokenizers/
    └── fineweb_1024_bpe.model
```

### Mode 2: Decode to Text

Decodes binary tokens back to text so you can re-tokenize with your own tokenizer.

```bash
python -m nanollm.dataset --mode decode
```

**Output**:
```
<nanollm_dir>/data/
├── datasets/fineweb10B_sp1024/   # Original binary (unchanged)
└── text/fineweb10B_sp1024/
    ├── fineweb_train_00000.jsonl
    ├── fineweb_train_00001.jsonl
    └── ...
```

---

## 4. Workflow for Training with rustbpe

### Option A: Use Pre-tokenized Data (Original)

1. Download binary:
   ```bash
   python -m nanollm.dataset --train-shards 80 --val-shards 1
   ```

2. Train using the included SentencePiece tokenizer:
   - nanollm will load `tokenizers/fineweb_1024_bpe.model`
   - Data is already tokenized as uint16 in .bin files

### Option B: Train with Custom Tokenizer (rustbpe)

1. Download binary:
   ```bash
   python -m nanollm.dataset --train-shards 80 --val-shards 1
   ```

2. Decode to text:
   ```bash
   python -m nanollm.dataset --mode decode
   ```

3. Use the text in `data/text/fineweb10B_sp1024/` to train your rustbpe tokenizer

4. Tokenize the text with rustbpe and create your own binary shards

---

## 5. Argparse Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `download` | Mode: `download` or `decode` |
| `--source` | `parameter-golf` | Dataset source (also supports `karpathy`) |
| `--variant` | `sp1024` | Tokenizer variant |
| `--train-shards` | `80` | Number of training shards |
| `--val-shards` | `1` | Number of validation shards |

---

## 6. Key Functions

| Function | Purpose |
|----------|---------|
| `download_parameter_golf_data()` | Download pre-tokenized binary files |
| `download_file_hf()` | Download single file via huggingface_hub |
| `get_tokenizer_path()` | Get path to tokenizer file |
| `decode_bin_to_text()` | Decode binary tokens to text |
| `list_bin_files()` | List available binary shard files |
| `load_bin_shard()` | Load a binary shard as numpy array |

---

## 7. Notes

- The tokenizer must match: using `fineweb_1024_bpe.model` to decode ensures we get the original text that was tokenized
- The binary files use uint16 (2 bytes per token) format
- HF_TOKEN environment variable is automatically used if set
- Both download and decode skip existing files to avoid re-downloading

---

## 8. Future Improvements (Optional)

- [ ] Add manifest-based validation (like original parameter-golf script)
- [ ] Support for other variants (byte260, different vocab sizes)
- [ ] Parallel download of shards
- [ ] Progress bars for downloads
- [ ] Verify shard integrity after download
