import os
import json
import random
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from nanollm.commons import get_base_dir


class HubDataset:

    def __init__(self,table,permutation=None):
        self.table=table
        self.permutation=permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self,seed):
        self.permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table,self.permutation)

    def __getitem__(self, index):
        physical_index= index if self.permutation is None else int(self.permutation[index])
        row = {column: self.table[column][physical_index].as_py() for column in self.table.column_names}
        return row


def load_hub_dataset( repo_id, subset= "default", split= "train"):
    base_dir=get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir= os.path.join(base_dir, "task_data", slug, subset, split)

    manifest_path=os.path.join(shards_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):

            if not os.path.exists(manifest_path):
                listing_url = f"https://huggingface.co/api/datasets/{repo_id}/parquet/{subset}/{split}"

                with urllib.request.urlopen(listing_url) as response:
                    shard_urls = json.loads(response.read())

                filenames = []

                for shard_index,shard_url in enumerate(shard_urls):
                    filename=f"{shard_index:05d}.parquet"
                    print(f"Downloading {shard_url}...")
                    with urllib.request.urlopen(shard_url) as response:
                        content = response.read()
                    with open(os.path.join(shards_dir, filename), "wb") as f:
                        f.write(content)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubDataset(table)


class Task:
    def __init__(self,start=0, stop=None, step=1):
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self,index):
        raise NotImplementedError

    def __len__(self):
        start= self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step -1) //step

        assert num >=0, f"Nefative number of examples???: {num}"
        return num

    def __getitem__(self, index:int):
        assert isinstance(index,int), f"Index must be an integet, got {type(index)}"
        physical_index = self.start + index *self.step
        conversation = self.get_example(physical_index)
        return conversation


    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)

        self.tasks=tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

        self.index_map=[]

        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))

        rng = random.Random(42)
        rng.shuffle(self.index_map)


    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0<= index < self.num_conversations, f"Idex {index} out of range for mixture with {self.num_conversations} conversations."
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks=tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum (self.lengths)


    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question,letters, choices):
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters,choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query

if __name__=="__main__":
    from tasks.mmlu import MMLU

    ds = MMLU( subset= "all", split = "auxiliary_train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="all", split="auxiliary_train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])


    print ("They Match: ", ex == ds[0])