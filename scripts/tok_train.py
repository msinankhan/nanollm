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



t0=time.time()
tokenizer=RustBPETokenizer(text_iter,args.vocab_size)
t1=time.time()

total_time=t1-t0

print(f"Time taken to train: {total_time:.2f}")


base_dir=get_base_dir()
tokenizer_dir=os.path.join(base_dir,"tokenizer_dir")

tokenizer.save(tokenizer_dir)


test_text= """Hello world, this is a test!
Numbers: 12, 223, 4932
Special characters: @%#&!*
Unicode: 你好世界 🌍
Contractions: I'm, you're, it's"""

encode=tokenizer.encode(test_text)
decode=tokenizer.decode(encode)

assert decode==test_text



vocab_size=tokenizer.get_vocab_size()
special_set=set(tokenizer.get_special_tokens)

token_string=[tokenizer.decode([token_id]) for token_id in range(vocab_size)]

token_bytes=[]

for token_id in range(vocab_size):
    token_str=token_string[token_id]

    if token_str in special_set:
        token_bytes.append(0)

    else:
        id_bytes=len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)


token_bytes=torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path=os.path.join(tokenizer_dir, "token_bytes.pt")

with open(token_bytes_path, 'wb') as f:
    torch.save(token_bytes,f)


print(f"Saved token_bytes to {token_bytes_path}")



from nanollm.report import get_report

token_bytes_nonzero=(token_bytes[token_bytes>0]).to(dtype=torch.int32)

get_report().log(section="Tokenization training", data=[
     vars(args), # argparse command line arguments
    {"train_time": total_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])