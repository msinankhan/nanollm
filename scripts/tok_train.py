import os
import time
from nanollm.commons import get_base_dir
import argparse
import torch
from nanollm.tokenizer import RustBPETokenizer
from nanollm.dataset import parquet_iter_batches


parser=argparse.ArgumentParser(description="Train a BPE tokenizer.")
parser.add_argument('--max_chars',type=int, default=10_000_000_000,help="Max characters to train on. (Default=10B)")
parser.add_argument('--doc_cap',type=int, default=10_000, help="Maximum characters per document. (Default =10k)")
parser.add_argument('--vocab_size', type=int, default=65536, help="Vocab Size. (Default: 2^16=65536) ")

args=parser.parse_args()

print(f"Vocab-Size: {args.vocab_size:,}")
print(f"Max Characters: {args.max_chars:,}")
print(f"Document Cap: {args.doc_cap:,}")


def text_iterator():
    nchars=0
    for batch in parquet_iter_batches(split="train"):
        for doc in batch:
            doc_text=doc

            if len(doc_text)>args.doc_cap:
                doc_text=doc_text[:args.doc_cap]

            nchars+=len(doc_text)
            yield doc_text

            if nchars>args.max_chars:
                return
            

text_iter=text_iterator()



