# SEt the working dire
import os
from pathlib import Path
import numpy as np
import tqdm
# os.chdir('/home/aiops/zhuty/tinyllama')
# add to PYTHONPATH
import sys
import json

# sys.path.append('/home/aiops/zhuty/tinyllama')
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt import Tokenizer
upper_limit = 3000000


ds_name = sys.argv[1]
# replace with your own working directory
BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/sample_processed/{ds_name}/"
output_file = f"/home/aiops/zhuty/tinyllama/processing/corpus_analysis/eos_locations_{ds_name}_sample{upper_limit}.json"

if '8k' in ds_name:
    length=8192
elif '32k' in ds_name:
    length=32768
else:
    raise ValueError("Invalid dataset name")

all_filenames = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.startswith("train_")]
print("Number of files:", len(all_filenames))
filenames = all_filenames

split = "train"
dataset = PackedDataset(
    filenames,
    # n_chunks control the buffer size.
    # Note that the buffer size also impacts the random shuffle
    # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
    n_chunks=512 if split == "train" else 2,
    block_size= length + 1,
    shuffle=False,
    seed=42,
    num_processes=1,
    process_rank=0,
    merge_method= "no",
    mask_attn = "",
    wrap=False
)



# load llama tokenizer
tokenizer_path = Path('./../tokenizer')
tokenizer = Tokenizer(tokenizer_path)
print("Tokenizer EOS token:", tokenizer.eos_id, "Token is:", tokenizer.decode(np.array([tokenizer.eos_id])) )
print("Tokenizer BOS token:", tokenizer.bos_id)
print("token id 13 token is ", tokenizer.decode(np.array([13])))

EOS_TOKEN = tokenizer.eos_id
print("EOS token:", EOS_TOKEN)


eos_locations = []

for i, item in tqdm.tqdm(enumerate(dataset), desc = "Processing"):
    # print("Item keys", item.keys())
    item = item['idx']
    if i >= upper_limit:
        break

    # Find the indices where EOS token occurs
    eos_indices = np.where(item == EOS_TOKEN)[0]
    # print("EOS indices:", eos_indices)
    eos_locations.append(eos_indices.tolist())

assert len(eos_locations) == upper_limit
# save as json, replace with your own
print("Saving to", output_file)
with open(output_file, 'w') as f:
    json.dump(eos_locations, f)