# Colab G4 reference training

This repository's reference run is a single-GPU d24 model trained in BF16 with
an `SSSL` attention pattern and 12 tokens per scaling parameter. The launcher is
`base_train.sh`; it deliberately separates preparation, pretraining, and
post-training so a wall-clock exit cannot start SFT from an incomplete base
model.

## Complete Colab CLI setup

The official `google-colab-cli` currently supports Linux and macOS. Commands in
this section that begin with `colab` run on your **local machine**. Commands
inside the SSH blocks run on the **Colab VM**.

The CLI documentation is at
<https://github.com/googlecolab/google-colab-cli>. We explicitly select OAuth2
below instead of relying on whichever authentication default a particular CLI
release uses.

### A. One-time local installation and authentication

Run locally, before allocating a runtime:

```bash
# Install or upgrade the official CLI in an isolated uv tool environment.
uv tool install -U google-colab-cli

# Confirm the executable and inspect its current command surface.
colab version
colab --help

# colab ssh accepts Ed25519 or ECDSA keys. Create an Ed25519 key once if needed.
command -v ssh
if [[ ! -f "$HOME/.ssh/id_ed25519" ]]; then
    ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519"
fi


# Trigger browser authentication without allocating a VM.
# Follow the URL, sign into the Colab Pro+ account, and paste the code back.
colab --auth oauth2 sessions

# Record the CU balance before doing anything billable.
colab --auth oauth2 usage
```

If `uv` is not installed locally, install it first from
<https://docs.astral.sh/uv/getting-started/installation/>. Do not paste OAuth
codes, GitHub tokens, or W&B keys into this repository or a command saved in
shell history.

Before creating either runtime, commit and push the exact code that will be
trained. A new Colab VM is ephemeral and must be able to retrieve that commit:

```bash
cd /home/sinan/Documents/nanollm
git status
git rev-parse HEAD
git push origin main
```

The repository URL used below is:

```text
https://github.com/msinankhan/nanollm.git
```

If the repository is private, authenticate interactively with `gh auth login`
inside the VM and use `gh repo clone msinankhan/nanollm`. Never put a personal
access token directly into the clone URL.

### B. CPU session: prepare all persistent assets

Create a named CPU session locally. This stage does not request a GPU and does
not consume G4 units:

```bash
colab --auth oauth2 new -s nanochat-prepare
colab --auth oauth2 status -s nanochat-prepare
colab --auth oauth2 drivemount -s nanochat-prepare /content/drive
colab --auth oauth2 ssh -s nanochat-prepare
```

Inside the CPU VM:

```bash
set -Eeuo pipefail
cd /content
git clone https://github.com/msinankhan/nanollm.git
cd nanollm

# Confirm that this is the intended, pushed training commit.
git rev-parse HEAD
git status --short

# The prepare action uses the CPU PyTorch dependency set.
python -m pip install --upgrade uv
./base_train.sh prepare
./base_train.sh status
exit
```

Back on the local machine, stop the CPU runtime and verify it was released:

```bash
colab --auth oauth2 stop -s nanochat-prepare
colab --auth oauth2 sessions
colab --auth oauth2 usage
```

Do not continue until Drive contains the tokenizer, task data, evaluation
bundle, and all 170 FineWeb-Edu shards under:

```text
/content/drive/MyDrive/nanollm-runs/reference-d24-r12/
```

### C. Allocate and inspect the G4 runtime

Run locally only when you are ready for compute-unit consumption to begin:

```bash
colab --auth oauth2 usage
colab --auth oauth2 new -s nanochat-g4 --gpu G4
colab --auth oauth2 status -s nanochat-g4
colab --auth oauth2 drivemount -s nanochat-g4 /content/drive
colab --auth oauth2 ssh -s nanochat-g4
```

Inside the G4 VM:

```bash
set -Eeuo pipefail
cd /content
git clone https://github.com/msinankhan/nanollm.git
cd nanollm
git rev-parse HEAD
git status --short

python -m pip install --upgrade uv

# Install the GPU environment, copy persistent shards to local NVMe, and verify
# the exact shard count and tokenizer vocabulary before training.
./base_train.sh hydrate

# Inspect the actual allocation rather than assuming the requested GPU arrived.
nvidia-smi
.venv/bin/python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("PyTorch:", torch.__version__)
PY

# Authenticate through W&B's hidden interactive prompt. The key is not placed
# in the repository or command line. Skip this only if WANDB_RUN=dummy.
.venv/bin/wandb login

./base_train.sh status
./base_train.sh preflight
```

The preflight must report one RTX PRO 6000-class GPU, compute capability
`(12, 0)`, BF16, and the `fa2_hub` attention backend. Do not start the long run
if it reports SDPA, the wrong tokenizer size, missing shards, NaN loss, or an
unexpected GPU.

### D. Start training in a persistent terminal

Use `tmux` inside the SSH session so a local terminal or network disconnect does
not send a hangup signal to training:

```bash
cd /content/nanollm
tmux new -s nanochat-train
```

Inside tmux:

```bash
set -o pipefail
cd /content/nanollm
./base_train.sh pretrain 2>&1 | tee /content/nanochat-pretrain.log
exit_code=${PIPESTATUS[0]}
./base_train.sh status
echo "pretrain exit code: $exit_code"
```

Detach without stopping training with `Ctrl-b`, then `d`. Reconnect later with:

```bash
# Local machine
colab --auth oauth2 status -s nanochat-g4
colab --auth oauth2 ssh -s nanochat-g4

# Colab VM
tmux attach -t nanochat-train
```

Exit code `75` means the wall-clock guard saved a valid checkpoint but the stage
is unfinished. Exit code `0` means the stage's validated completion marker was
published. Any other nonzero code is a failure that must be investigated before
resuming.

### E. Continue after Colab reclaims a VM

Allocate a new session, mount Drive, clone the same commit, and then run:

```bash
cd /content/nanollm
python -m pip install --upgrade uv
./base_train.sh hydrate
./base_train.sh pretrain
```

`pretrain` automatically selects the newest fully committed, checksum-valid
checkpoint. After base training completes, use a fresh G4 session and replace
the last command with:

```bash
./base_train.sh posttrain
```

The same `posttrain` command resumes interrupted SFT and ChatRL stages.

### F. Stop billing deliberately

Never stop the session while a checkpoint says it is still publishing. Once the
stage has exited and `./base_train.sh status` has returned, run locally:

```bash
colab --auth oauth2 status -s nanochat-g4
colab --auth oauth2 usage
colab --auth oauth2 stop -s nanochat-g4
colab --auth oauth2 sessions
colab --auth oauth2 usage
```

The CLI stores local session metadata under `~/.config/colab-cli/`. `colab stop`
is the operation that releases the assigned VM; closing the SSH window is not.

## Storage layout


- `NANOLLM_BASE_DIR`: persistent tokenizer, task data, checkpoints, and reports.
  On Colab this should live in mounted Google Drive.
- `NANOLLM_PERSISTENT_DATA_DIR`: persistent copy of the 170 FineWeb-Edu shards.
- `NANOLLM_DATA_DIR`: fast runtime-local copy used by training. The default is
  `/content/nanollm-data`.
- `NANOLLM_CHECKPOINT_STAGING_DIR`: fast runtime-local checkpoint staging. A
  checkpoint is checksummed here before it is published to persistent storage.

The defaults assume Drive is mounted at `/content/drive`. Mount Drive before
running `prepare`, `hydrate`, or any training stage.

## 1. Prepare without a GPU

Use a CPU runtime:

```bash
./base_train.sh prepare
```

This downloads 170 shards into persistent storage, trains the 32,768-token
tokenizer if it does not already exist, downloads the CORE/ChatCORE/SFT/RL
datasets, evaluates the tokenizer, and verifies both the shard count and
vocabulary size. Do not switch runtimes until this finishes.

If the static dataset is stored as a private Kaggle dataset instead, download
and extract it into `NANOLLM_PERSISTENT_DATA_DIR`; the remaining commands are
unchanged. Kaggle is suitable for versioned static input data. Checkpoints
remain in Drive because they change throughout training.

## 2. Hydrate local storage

After switching to the G4 runtime:

```bash
./base_train.sh hydrate
```

This copies only missing files from persistent storage to `/content` and then
validates the local dataset. Subsequent sessions repeat this step before
resuming training.

## 3. Blackwell preflight

Start with device batch 16:

```bash
./base_train.sh preflight
```

Then test 32 using a different preflight tag:

```bash
MODEL_TAG=d24-r12-bf16-reference-b32 DEVICE_BATCH_SIZE=32 ./base_train.sh preflight
```

Keep batch 32 only if it has comfortable memory headroom and better steady-state
tokens/second. The reference run does not enable the custom FP8 path.

## 4. Pretrain and resume

```bash
./base_train.sh pretrain
```

The launcher automatically resumes the newest checksummed checkpoint. It exits
with status 75 when the current Colab session ends safely but the complete
training horizon has not yet been reached. Rehydrate a new runtime and execute
the same command again.

The post-training phase is unlocked only after
`base_checkpoints/<model-tag>/training.complete.json` exists and validates its
referenced checkpoint.

## 5. SFT and ChatRL

Use a fresh G4 session after base training completes:

```bash
./base_train.sh posttrain
```

This runs full base evaluation, full SFT, SFT ChatCORE evaluation, ChatRL, and
the final RL ChatCORE evaluation. SFT and ChatRL save their optimizer and exact
next-data position, obey the same wall-clock guard, and automatically resume
from their newest checksummed checkpoint. Completed training and evaluation
stages have separate markers and are skipped if the command is restarted.

If `posttrain` exits with status 75, start a fresh G4 session and run the same
command again:

```bash
./base_train.sh posttrain
```

For SFT, the saved state includes the best-fit packing cursor and conversation
buffer before the pending batch. For ChatRL, it includes the optimizer,
`next_step`, and the corresponding GSM8K example offset.

## Status and overrides

```bash
./base_train.sh status
```

All important settings are environment overrides. For example:

```bash
WANDB_RUN=d24-reference DEVICE_BATCH_SIZE=16 ./base_train.sh pretrain
```

Never change depth, token ratio, vocabulary, attention pattern, global batch,
SFT/RL batch settings, or model tag while resuming the same run. Use a new model
tag for a new recipe.
