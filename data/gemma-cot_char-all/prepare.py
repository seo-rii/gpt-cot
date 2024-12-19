"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import string
import requests
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset


num_proc=16
num_proc_load_dataset=16

# create a mapping from characters to integers

chars = string.printable

stoi = { str(ch):i for i,ch in enumerate(chars) }
itos = { i:str(ch) for i,ch in enumerate(chars) }

# add eot

stoi['<eot>']=100
itos[100]='<eot>'


print(stoi)
print(itos)

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


math_dataset = load_dataset("json", data_files="simpleMath4.json", split="train", num_proc=num_proc_load_dataset)
math_dataset = Dataset.from_list(math_dataset)
math_dataset = math_dataset.rename_column("query", "text")

logic_dataset = load_dataset("json", data_files="prontoqa_merged.json", split="train", num_proc=num_proc_load_dataset)
logic_dataset = Dataset.from_list(logic_dataset)

# Process logic dataset
logic_dataset = logic_dataset.remove_columns(["question"])  # Remove 'question' column
logic_dataset = logic_dataset.map(lambda x: {"correct": 1 if x["correct"] == "True" else 0})  # Convert 'correct' column to 1/0
logic_dataset = logic_dataset.rename_column("query", "text")

openweb_dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

dataset = Dataset.from_dict({
    'text': math_dataset['text'] + logic_dataset['text'] + openweb_dataset['text'],
})
split_dataset = dataset.train_test_split(test_size=0.1, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val


def process(example):
    ids = encode(example['text']) # encode_ordinary ignores any special tokens
    ids.append(100) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()



# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': len(chars)+1,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
