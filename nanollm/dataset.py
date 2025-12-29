import os
import time
import argparse
import requests 
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanollm.commons import get_base_dir



BASE_URL="https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARDS=1822

index_to_filename=lambda index:f"shard_{index:05d}.parquet"
base_dir=get_base_dir()
DATA_DIR=os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR,exist_ok=True)


def list_parquet_files(data_dir=None):
    data_dir=DATA_DIR if data_dir is None else data_dir

    parquet_files=sorted([
        f for f in data_dir if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    parquet_paths=[os.path.join(data_dir,f) for f in parquet_files]

    return parquet_paths


def parquet_iter_batches(split,start=0,step=1):

    assert split in ["train" , "val"], "Invalid split, must be 'train' or 'val'."
    parquet_paths=list_parquet_files()
    parquet_paths=parquet_paths[:-1] if split=='train' else parquet_paths[-1:]

    for file in parquet_paths:
        pf=pq.ParquetFile(file)
        for rg_idx in range (start,pf.num_row_groups,step):
            rg=pf.read_row_group(rg_idx)
            texts=rg.column('text').to_pylist()
            yield texts



def download_single_file(index):
    filename=index_to_filename(index)
    file_path=os.path.join(DATA_DIR,filename)

    if os.path.exists(file_path):
        print(f"Skipping {file_path} as it already exists.")
        return True
    
    url=f"{BASE_URL}/{filename}"
    print(f"Downloading file {filename}")

    max_attempts=5

    for attempt in range(1,max_attempts+1):
        try:
            response=requests.get(url=url,stream=True,timeout=30)
            response.raise_for_status()


            temp_path=filename +f".tmp"

            with open(temp_path,'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)

            os.rename(temp_path,filename)

            print(f"Sucessfully Downloaded {filename}")

            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename} : {e}")

            for path in [file_path +f".tmp",file_path]:
                if os.path.exists:
                    try:
                        os.remove(path)
                    except:
                        pass


            if attempt<max_attempts:
                wait_time=2**attempt
                print(f"Wating {wait_time} seconds before retrying Download.")
                time.sleep(wait_time)

            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Download dataset for pretraining.")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download, -1=disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of Parallel download workers (default:4)")

    args=parser.parse_args()




    num=MAX_SHARDS+1 if args.num_files==-1 else min(args.num_files, MAX_SHARDS+1)
    ids_to_download=list(range(num))

    print(f"Downlading {len(ids_to_download)} shards using {args.num_workers} workers...")

    print(f"Target Directory{DATA_DIR}")

    print()

    with Pool(processes=args.num_workers) as pool:
        results= pool.map(download_single_file, ids_to_download)


    successful= sum(1 for success in results if success)

    print(f"Done! Downloaded:{successful}/{len(ids_to_download)} shards to {DATA_DIR}")