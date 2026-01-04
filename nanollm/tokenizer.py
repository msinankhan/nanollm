import os
import copy
from functools import lru_cache


SPECIAL_TOKENS= [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    def __init__(self, enc, bos_token):
        self.enc=enc
        self.bos_token_id=self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer=rustbpe.Tokenizer()
        vocab_size_no_special= vocab_size-len(SPECIAL_TOKENS)

        assert vocab_size_no_special >=256, f"vocab_size_no_special should be atleast 256 but got {vocab_size_no_special}"

        tokenizer.train_from_iterator(text_iterator,vocab_size_no_special,pattern=SPLIT_PATTERN) #THIS LINE IS WHERE TRAINING HAPPENS.

        pattern=tokenizer.get_pattern() # Rust is using fancy_regex, which: **May** internally normalize behavior and **May** differ subtly in Unicode handling. The effective pattern is the one Rust accepted and stored.
        mergeable_ranks_list=tokenizer.get_mergeable_ranks()

        mergeable_ranks={bytes(k):v for k,v in mergeable_ranks_list}

        tokens_offset=len(mergeable_ranks)

        special_tokens={name:tokens_offset+i for i, name in enumerate(SPECIAL_TOKENS)}


        # The following is a deterministic byte-level BPE executor
        enc=tiktoken.Encoding(                               #This part constructs the final, frozen tokenizer used at inference time.
            name="rustbpe",                                  #It is equivalent to loading a model’s weights into a runtime.
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, #This is the entire learned vocabulary.
            special_tokens=special_tokens
        )

        return cls(enc,"<|bos|>")