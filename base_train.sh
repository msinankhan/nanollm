export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/disk2/Sinan/nanollm/.cache/nanochat"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WANDB_RUN=26
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NANOCHAT_DTYPE=bfloat16
mkdir -p $NANOCHAT_BASE_DIR
uv sync --extra gpu
source .venv/bin/activate

# if [ -z "$WANDB_RUN" ]; then
#     # by default use "dummy" : it's handled as a special case, skips logging to wandb
#     WANDB_RUN=dummy
# fi

python -m nanollm.report reset


# python -m nanollm.dataset -n 8

# python -m nanollm.dataset -n 170 &

# DATASET_DOWNLOAD_PID=$!

# python -m scripts.tok_train

# python -m scripts.tok_eval



# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID



torchrun  --nproc_per_node=4 -m scripts.base_train -- --depth=24 --target-params-data-ratio=20 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
# torchrun  --nproc_per_node=2 -m scripts.base_eval -- --device-batch-size=16

