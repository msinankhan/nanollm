"""Download every evaluation, SFT, and RL artifact before allocating a GPU."""

import os
import shutil
import tempfile
import zipfile

from nanollm.commons import download_file_with_lock, get_base_dir
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Extract or repair the base-model evaluation bundle without GPU imports."""
    eval_bundle_dir = os.path.join(get_base_dir(), "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        shutil.copytree(
            os.path.join(tmpdir, "eval_bundle"),
            eval_bundle_dir,
            dirs_exist_ok=True,
        )
    print(f"Eval bundle placed at {eval_bundle_dir}")


def main():
    bundle_path = download_file_with_lock(
        EVAL_BUNDLE_URL,
        "eval_bundle.zip",
        postprocess_fn=place_eval_bundle,
    )
    # Always merge the bundle so a partially deleted directory is repaired.
    place_eval_bundle(bundle_path)

    datasets = [
        SmolTalk(split="train"),
        SmolTalk(split="test"),
        MMLU(subset="all", split="auxiliary_train"),
        MMLU(subset="all", split="test"),
        GSM8K(subset="main", split="train"),
        GSM8K(subset="main", split="test"),
        ARC(subset="ARC-Easy", split="test"),
        ARC(subset="ARC-Challenge", split="test"),
        HumanEval(),
    ]
    print("Post-training data ready:")
    for dataset in datasets:
        print(f"  {type(dataset).__name__}: {len(dataset):,} examples")


if __name__ == "__main__":
    main()
