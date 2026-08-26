import torch
import pyarrow.parquet as pq


from nanollm.commons import get_dist_info
from nanollm.dataset import list_parquet_files

def _document_batches(split,resume_state_dict, tokenizer_batch_size):
    ddp,ddp_rank,ddp_local_rank ,ddp_world_size=get_dist_info()

    parquet_paths=list_parquet_files()

    assert (len(parquet_paths))!=0, f"NO dataset parquet files found, did you run dataset.py?"

    parquet_paths=parquet_paths[:-1] if split=="train" else parquet_paths[-1:]

    resume_pq_idx=resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx=resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch=resume_state_dict.get("epoch", 1) if resume_state_dict else 1

    first_pass=True # This is to accomodate special handling on Startup/resume. After that, we loop normally. 
    pq_idx=resume_pq_idx # This jumps you up to the right file. 
    epoch=resume_epoch

    while True:

        pq_idx=resume_pq_idx if first_pass else 0 # We need to reset after every epoch. 

        while pq_idx<len(parquet_paths):
            file_path=parquet_paths[pq_idx]

            pf=pq.ParquetFile(file_path)

            if first_pass and (resume_rg_idx is not None) and (pq_idx==resume_pq_idx):
                base_idx=resume_rg_idx//ddp_world_size +1
                rg_idx=base_idx*ddp_world_size+ddp_rank

                if rg_idx>=pf.num_row_groups:
                    pq_idx+=1
                    continue
                resume_rg_idx=None

            else:
                rg_idx=ddp_rank

            while rg_idx<pf.num_row_groups:
                rg=pf.read_row_group(rg_idx)
                batch=rg.column('text').to_pylist()

                for i in range(0,len(batch),tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx,rg_idx, epoch)

                rg_idx+=ddp_world_size
            pq_idx+=1

        first_pass=False
        epoch+=1
        
def tokenizing_distributed_data_loader_with_state(tokenizer, B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):

    assert split in ['train','val'], f"Split should be either train or val :{split}"

    batches=_document_batches (split,resume_state_dict,tokenizer_batch_size)
    needed_tokens=B*T+1
    bos_token=tokenizer.get_bos_token_id()
    token_buffer=[]
    pq_idx,rg_idx, epoch=0,0,1

    while True:

        while len(token_buffer)< needed_tokens:
            doc_batch,(pq_idx,rg_idx,epoch) = next(batches)
            tokens_list=tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads )

            for tokens in tokens_list:
                token_buffer.extend(tokens) #When we flatten the buffer, the document boundaries are lost and tokens become one continuous stream.
                                            #BOS is no longer guaranteed at sequence start
                                            #Attention can span across unrelated documents.

        tokens=token_buffer[:needed_tokens] # We grab just enough tokens for inputs, targets. 
        token_buffer=token_buffer[B*T:]     # We overlap by 1 token, this extra token belongs to the target

        use_cuda=torch.device(device).type=="cuda"

        scratch=torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        inputs=scratch[:-1].view(B,T).to(device=device, non_blocking=use_cuda)
        targets=scratch[1:].view(B,T).to(device=device,non_blocking=use_cuda)

        yield inputs,targets, {"pq_idx":pq_idx, "rg_idx":rg_idx, "epoch":epoch}


def tokenizing_distributed_data_loader(*args,**kwargs):
    """A Helper function to return inputs and targets without the state dictionary."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args,**kwargs):
        yield inputs, targets


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, B,T, split,
        tokenizer_threads=4, tokenizer_batch_size=128,
        device="cuda", resume_state_dict=None,
        buffer_size=1000
):
    assert split in ['train', "val"], f"Split should be either train or val: {split}"

    row_capacity=T+1
    doc_buffer=[]
    batches=_document_batches(split,resume_state_dict, tokenizer_batch_size)
    bos_token=tokenizer.get_bos_token_id()
    pq_idx,rg_idx,epoch=0,0,1

    def refill_buffer():
        nonlocal pq_idx,rg_idx,epoch
        doc_batch, (pq_idx,rg_idx,epoch)= next(batches)
        token_lists=tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    while True:
        rows=[]
        for _ in range(B):
            row=[]
            while len(row) < row_capacity:
                while len(doc_buffer)< buffer_size:
                    refill_buffer() # We fill the doc_buffer with 999 document's tokens

                remaining=row_capacity-len(row)


                best_idx=-1
                best_len=0

                for i, doc in enumerate(doc_buffer): # Here, we are trying to figure out which is the longest full document that can fit in the row[], without us having to trim it.
                    if len(doc)<=remaining and len(doc)>best_len:
                        best_idx=i
                        best_len=len(doc)

                if best_idx>=0:                     #If we find such an index where the len of the doc is lesser than the the row capacity, we append it to the row[].
                    row.extend(doc_buffer.pop(best_idx))

                else:
                    """This address the case where every document is bigger than the remaining capacity, for which we choose the shortest document to add into the list and crop the rest that doesn't fit in the row_capacity.
                     It also prevents token loss, when we choose a smaller document, as if we must dicard documents we will discard as few tokens as possible. """
                    shortest_idx=min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i])) # We find the shortest document to crop, so that it may be as full as possible in the row. 
                    doc=doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])

            rows.append(row[:row_capacity])

        use_cuda=torch.device(device).type=="cuda"

        batch_tensor=torch.tensor(rows,dtype=torch.long, pin_memory=use_cuda)
        inputs=batch_tensor[:,:-1].to(device=device, non_blocking=use_cuda)
        targets=batch_tensor[:,1:].to(device=device, non_blocking=use_cuda) #TODO:cpu_buffer, gpu_buffer.

        yield inputs, targets, {"pq_idx":pq_idx, "rg_idx":rg_idx, "epoch":epoch}


def tokenizing_distributed_data_loader_with_bos_bestfit(*args,**kwargs):
    """Helper function that omits the state dictionary from yielded batches."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args,**kwargs):
        yield inputs, targets
