import torch
import pyarrow.parquet as pq


from nanollm.commons import get_dist_info
from nanollm.dataset import list_parquet_files

def _document_batches(split,resume_state_dict, tokenizer_batch_size):
    ddp,ddp_rank,ddp_local_rank ,ddp_world_size=get_dist_info()

    parquet_paths=list_parquet_files()

    assert (len(parquet_paths))!=0, f"NO dataset parquet files found, did you run dataset.py?"

    parquet_paths=parquet_paths[:-1] if split=="train" else parquet_paths[-1:]

    resume_pq_idx=resume_state_dict["resume_pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx=resume_state_dict["resume_rg_idx"] if resume_state_dict is not None else None 
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

                if rg_idx>pf.num_row_groups:
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

                rg_idx+=1
            pq_idx+=1

        first_pass=False
        epoch+=1
        
