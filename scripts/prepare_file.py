import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

# Filename for SlimPajama
slimpajama_sets = {
    "train": "train/*",
    "valid": "valid/*",
}

REMOVE_WEIRD = False


def prepare_full(
        source_path: Path,
        tokenizer_path: Path,
        destination_path: Path,
        chunk_size: int,
        shortname: str = "ind",
        split: str = "train",
        filenames_subset: List[str] = None,
        process_id: int = 0,
        text_key: str = "text",
        char_percent: float = 0,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset

    if not filenames:
        raise RuntimeError(
            f"No files matching {slimpajama_sets[split]} found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_{shortname}_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )
    print("Remove Weird", REMOVE_WEIRD)
    for filepath in filenames:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, row in enumerate(tqdm(f, desc=f"Processing {filepath}, char_percent={char_percent}")):
                text = json.loads(row)[text_key]
                if REMOVE_WEIRD:
                    text = text.replace("\\n", "\n")
                if char_percent == 0 or (char_percent > 0 and np.random.rand() > char_percent):  # normal encode
                    text_ids = tokenizer.encode(text)
                else:
                    # encode as characters
                    # print("Encoding as characters for row", i)
                    text_ids = tokenizer.encode(text, bpe_dropout=1)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # if split == "train":
    #     builder.write_reminder()
    # else:
    #     print("Skipping reminder file for the validation set")
    # just skip as this might cause issues when chunk size is large

def prepare(
        source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
        tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
        destination_path: Path = Path("data/red_pajama_sample"),
        short_name: str = "ind",
        chunk_size: int = 2049 * 1024,
        split: str = "train",
        percentage: float = 1.0,
        text_key: str = "text",
        char_percent: float = 0,
) -> None:
    import time

    filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
    filenames = filenames[:int(len(filenames) * percentage)]

    # num_processes = min(cpu_count(), len(filenames))
    if split == 'train':
        num_processes = 32
    if split == "valid":
        num_processes = 10  # make a small
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(
        source_path, tokenizer_path, destination_path, chunk_size, short_name, split, list(subset), i, text_key,
        char_percent))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)