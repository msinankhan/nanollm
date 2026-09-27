"""Download every evaluation, SFT, and RL artifact before allocating a GPU."""

import os

from nanollm.commons import download_file_with_lock, get_base_dir
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk


def main():
    bundle_path = download_file_with_lock(
        EVAL_BUNDLE_URL,
        "eval_bundle.zip",
        postprocess_fn=place_eval_bundle,
    )
    if not os.path.isdir(os.path.join(get_base_dir(), "eval_bundle")):
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
