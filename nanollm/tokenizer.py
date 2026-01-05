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
    

    @classmethod
    def from_directory(cls,tokenizer_dir):
        pickle_path=os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, 'rb') as f:
            enc=pickle.load(f)
        return cls(enc, "<|bos|>")
    
    @classmethod
    def from_pretraining(cls, tiktoken_name):
        enc=tiktoken.get_encoding(tiktoken_name)

        return cls(enc,"<|endoftext|>")
    
    def get_vocab_size(self):
        return self.enc.n_vocab
    
    def get_special_tokens(self):
        return self.enc.special_tokens_set
    
    def id_to_token(self,id):
        return self.enc.decode([id])
    
    @lru_cache(maxsize=32)
    def encode_special(self,text):
        return self.enc.encode_single_token(text)

    def get_bos_token(self):
        return self.bos_token_id
    
    def encode(self,text,prepend=None, append=None, num_threads=8):

        if prepend is not None:
            prepend_id=prepend if isinstance(prepend,int) else self.encode_special(prepend)

        if append is not None:
            append_id=append if isinstance(append,int) else self.encode_special(append)


        if isinstance(text,str):
            ids=self.enc.encode_ordinary(text)

            if prepend is not None:
                ids.insert(0,prepend_id)

            if append is not None:
                ids.append(append_id)

        elif isinstance(text,list):
            ids=self.enc.encode_ordinary_batch(text,num_threads=num_threads)

            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0,prepend_id)

            elif append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)

        else:
            raise ValueError(f"Invalid Input type:{type(text)}")
        
        return ids
    

    def __call__(self, *args, **kwargs):
        return self.encode(*args,*kwargs)


    def decode(self,ids):
        return self.enc.decode(ids)
    
    def save(self,tokenizer_dir):
        os.makedirs(tokenizer_dir,exist_ok=True)
        pickle_path=os.path.join(tokenizer_dir,"tokenizer.pkl")

        with open(pickle_path,"wb") as f:
            pickle.dump(self.enc,f)

        print(f"Saved the tokenizer encoding to {pickle_path}")



    def render_conversation(self, conversation,max_tokens=2048):
        ids, mask=[],[]


        def add_tokens(token_ids,mask_val):
            if isinstance(token_ids,int):
                token_ids=[token_ids]

            ids.extend(token_ids)
            mask.extend([mask_val]*len(token_ids))


            if conversation["messages"][0]["role"]=="system":
                conversation=copy.deepcopy(conversation)
                messages=conversation["messages"]
                assert messages[1]["role"] =="user", "System message must be followed by a user message"
                messages[1]["content"]=messages[0]["content"] + "\n\n" + messages[1]["content"]
                messages=messages[1:]

            else:
                messages=conversation["messages"]


            assert len(messages) >=1, f"Conversation has less than 1 message: {len(messages)}"


            bos= self.bos_token_id()
            user_start,user_end=self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
            assistant_start,assistant_end=self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
            python_start,python_end=self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
            output_start,output_end=self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

            add_tokens(bos,0)

            for i, message in enumerate(messages):
                must_be_from="user" if i%2==0 else "assistant"
                assert message["role"]==must_be_from, f"Message {i} must be from {must_be_from} but should be from {message["role"]}"

                content=message["content"]
                

                if message["role"]=="user":
                    assert isinstance(content,str), f"User messages are simply expected to be string, but got {type(content)}"

                    add_tokens(user_start,0)
                    value_ids=self.encode(content)
                    add_tokens(value_ids,0)
                    add_tokens(user_end)


                elif message["role"]=="assistant":
                    add_tokens(assistant_start,0)

                    if isinstance(content,str):
                        value_ids=self.encode(content)
                        add_tokens(value_ids,1)

                    elif isinstance(content,list):
                        for part in content:
                            value_ids=self.encode(part["text"])

                            if part["type"]=="text":
                                add_tokens(value_ids,1)
                            elif part["type"]=="python":
                                add_tokens(python_start,1) #Gotta set the mask as 1 for the tokens that the model will train on. 
                                add_tokens(value_ids,1)
                                add_tokens(python_end,1)

                            elif part["type"]=="python_end":
                                add_tokens(output_start,0)
                                add_tokens(value_ids,0)
                                add_tokens(output_end,0)

                            else:
                                raise ValueError(f"Unknown part type passed in the conversation{part["type"]}")
                            
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                
                add_tokens(assistant_end)


            ids=ids[:max_tokens]
            mask=mask[:max_tokens]

            return ids,mask
        


    def visualize_tokenization(self,ids,mask,with_token_id=False):

        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'

        tokens=[]

        for i, (token_id,mask_val) in enumerate(zip(ids,mask)):
            token_str=self.decode([token_id])
            color=GREEN if mask_val ==1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}{token_id}{RESET}")

        return '|'.join(tokens)
    

    def render_for_completion(self,conversation):
        conversation=copy.deepcopy(conversation)
        messages=conversation["messages"]
        assert messages[-1]["role"]=="assistant", f"Last message must be from Assistant."
        messages.pop()

        ids,mask=self.render_conversation(conversation)

        assistant_start=self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids
    

    def get_tokenzier():
        from nanollm.commons import get_base_dir
        base_dir=get_base_dir()
        tokenizer_directory=os.path.join(base_dir,"tokenizer_dir")

        return RustBPETokenizer.from_directory(tokenizer_directory)
    
    def get_token_bytes(device="cpu"):
        import torch
        from nanollm.commons import get_base_dir

        base_dir=get_base_dir()
        tokenizer_dir=os.path.join(base_dir, "tokenizer")
        token_bytes_path=os.path.join(tokenizer_dir, "token_bytes.pt")

        assert os.path.exists(token_bytes_path), f" Token bytes not found at {token_bytes_path}? It gets written by tok_train.py."

        with open(token_bytes_path,'rb') as f:
            token_bytes=torch.load(f,map_location=device) #GPU memory is precious; loading static metadata like token bytes on GPU is wasteful.
        return token_bytes                                #Most of the time, you only need them for encoding/decoding, which is CPU-light.