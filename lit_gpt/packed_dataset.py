# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}

# def get_fragment_lens(chunk, skip_indices):
#     # adapted from https://github.com/yuzhaouoe/pretraining-data-packing/blob/8ee89732e73e9c5dec5af858289512206a050a0d/packing_dataset.py#L165
#     # need to calculate the fragment lengths for each chunk
#     chunk_size = len(chunk)
#     cur_fragment_lens = []
#     prev = 0
#     for token_idx, token in enumerate(chunk):
#         if token == 2 and token_idx not in skip_indices:
#             cur_fragment_lens.append(token_idx - prev + 1)
#             prev = token_idx + 1
#     if prev != chunk_size:
#         cur_fragment_lens.append(chunk_size - prev)
#     # print("Fragment lens:", cur_fragment_lens)
#     # print("Sum of fragment lens:", sum(cur_fragment_lens))
#     return cur_fragment_lens, len(cur_fragment_lens)

# Optimized function using NumPy
def get_fragment_lens_optimized(chunk, skip_indices):
    skip_indices_set = set(skip_indices)
    is_two = np.where(chunk == 2)[0]
    filtered_indices = np.array([idx for idx in is_two if idx not in skip_indices_set])
    # if len(skip_indices) > 0:
    #     print("Skipper indices:", len(skip_indices), "Filtered indices:", len(filtered_indices))
    # # Adjust how fragment lengths are calculated to match the original function
    if filtered_indices.size > 0:
        fragment_lengths = []
        prev = 0
        for idx in filtered_indices:
            fragment_lengths.append(idx - prev + 1)
            prev = idx + 1
        if prev < len(chunk):
            fragment_lengths.append(len(chunk) - prev)
    else:
        fragment_lengths = [len(chunk)]  # If no valid indices, the entire chunk is one fragment

    return fragment_lengths, len(fragment_lengths)


def get_fragment_lens_fixed_length(chunk, fixed_length, is_multiple=True):
    assert fixed_length > 0, "Fixed length must be greater than 0, but got {}".format(fixed_length)
    assert not is_multiple or len(chunk) % fixed_length == 0, "Chunk length must be a multiple of fixed length, but got {} and {}".format(len(chunk), fixed_length)
    filtered_indices = np.arange(fixed_length, len(chunk), fixed_length) -1 # -1 was added on Oct 17, before training of intradm models, but the old models are trained with the bug
    # if len(skip_indices) > 0:
    #     print("Skipper indices:", len(skip_indices), "Filtered indices:", len(filtered_indices))
    # # Adjust how fragment lengths are calculated to match the original function
    if filtered_indices.size > 0:
        fragment_lengths = []
        prev = 0
        for idx in filtered_indices:
            fragment_lengths.append(idx - prev + 1)
            prev = idx + 1
        if prev < len(chunk):
            fragment_lengths.append(len(chunk) - prev)
    else:
        fragment_lengths = [len(chunk)]  # If no valid indices, the entire chunk is one fragment

    return fragment_lengths, len(fragment_lengths)

def get_fragment_lens_fixed_length_intramask(chunk, fixed_length, is_multiple=True):
    assert fixed_length > 0, "Fixed length must be greater than 0, but got {}".format(fixed_length)
    assert not is_multiple or len(chunk) % fixed_length == 0, "Chunk length must be a multiple of fixed length, but got {} and {}".format(len(chunk), fixed_length)
    is_two = np.where(chunk == 2)[0]
    filtered_indices = is_two
    fixed_indices = np.arange(fixed_length, len(chunk), fixed_length)
    # gei the union
    filtered_indices = np.union1d(filtered_indices, fixed_indices)

    # if len(skip_indices) > 0:
    #     print("Skipper indices:", len(skip_indices), "Filtered indices:", len(filtered_indices))
    # # Adjust how fragment lengths are calculated to match the original function
    if filtered_indices.size > 0:
        fragment_lengths = []
        prev = 0
        for idx in filtered_indices:
            fragment_lengths.append(idx - prev + 1)
            prev = idx + 1
        if prev < len(chunk):
            fragment_lengths.append(len(chunk) - prev)
    else:
        fragment_lengths = [len(chunk)]  # If no valid indices, the entire chunk is one fragment

    return fragment_lengths, len(fragment_lengths)

def split_into_docs(chunk, eos_id = 2):
    """
    Split the chunk into documents based on the EOS token
    Args:
        chunk:
        eos_id:

    Returns: list of documents
    """
    eos_indices = np.where(chunk == eos_id)[0]
    docs = []
    start_idx = 0
    for eos_idx in eos_indices:
        docs.append(chunk[start_idx:eos_idx])
        start_idx = eos_idx + 1
    if start_idx < len(chunk):
        docs.append(chunk[start_idx:])
    return docs

# def calculate_bigram_overlap(prev, next):
#     prev_bigrams = set(zip(prev[:-1], prev[1:]))
#     next_bigrams = set(zip(next[:-1], next[1:]))
#     if len(prev_bigrams) == 0 or len(next_bigrams) == 0:
#         return 0
#     return len(prev_bigrams.intersection(next_bigrams)) / min(len(prev_bigrams), len(next_bigrams))

def calculate_bigram_set_overlap(prev_bigrams, next_bigrams):
    """
    Calculate the overlap between two sets of bigrams
    Args:
        prev_bigrams:
        next_bigrams:

    Returns:

    """
    if len(prev_bigrams) == 0 or len(next_bigrams) == 0:
        return 0
    return len(prev_bigrams.intersection(next_bigrams)) / min(len(prev_bigrams), len(next_bigrams))

def get_new_count(input_count, target_count_distribution):
    # first, filter out the counts larger than the input count
    # target_count_distribution = {k: v for k, v in target_count_distribution.items() if k <= input_count}
    # if len(target_count_distribution) == 0:
      #  return input_count
    # renormalize the distribution
    #total = sum(target_count_distribution.values())
    #target_count_distribution = {k: v/total for k, v in target_count_distribution.items()}

    # randomly sample from the target count distribution
    target_counts = list(target_count_distribution.keys())
    target_probs = list(target_count_distribution.values())
    new_count = np.random.choice(target_counts, p=target_probs)
    if new_count > input_count:
        return input_count
    else:
        return new_count


TARGET_COUNT_DISTRIBUTION = {4: {0: 1.0}, 5: {0: 1.0}, 6: {0: 1.0}, 7: {0: 0.4057439657806294, 1: 0.3525206232813932, 2: 0.2417354109379774}, 8: {2: 0.19703804966962862, 3: 0.47527910685805425, 4: 0.32768284347231713}, 9: {4: 0.2564597923206955, 5: 0.6882878531755614, 6: 0.05525235450374306}, 10: {6: 0.7427482956823954, 7: 0.2572517043176046}, 11: {7: 0.6765799256505576, 8: 0.32342007434944237}, 12: {8: 0.7440944881889764, 9: 0.2559055118110236}, 13: {9: 1.0}, 14: {9: 0.11928934010152284, 10: 0.8807106598984772}, 15: {10: 0.6507633587786259, 11: 0.34923664122137404}, 16: {11: 1.0}, 17: {11: 1.0}, 18: {11: 0.4, 12: 0.6}, 19: {12: 1.0}, 20: {12: 0.4098360655737705, 13: 0.5901639344262295}}
def get_random_skip_indices(chunk, eos_id):
    """
    Get random indices to skip
    Args:
        chunk:
        eos_id:
        skip_prob:

    Returns:

    """
    eos_indices = np.where(chunk == eos_id)[0]
    old_eos_count = len(eos_indices)
    if old_eos_count not in TARGET_COUNT_DISTRIBUTION:
        if old_eos_count < 4:
            old_eos_count = 4
        elif old_eos_count > 20:
            old_eos_count = 20

    new_count = get_new_count(old_eos_count, TARGET_COUNT_DISTRIBUTION[old_eos_count])
    skipped_count = len(eos_indices) - new_count
    if skipped_count == 0:
        return []
    skip_indices = np.random.choice(eos_indices, size=skipped_count, replace=False)
    return skip_indices

def get_eos_indices_between_relevant_docs(chunk, sim_func, eos_id, lower_bound, upper_bound):
    """
    Merge neighboring documents if they have similarity
    Args:
        chunk: an array of tokens
        eos_id: EOS token id used to mark the boundary of documents

    Returns: indices of the EOS tokens that need to be replaced
    """
    eos_indices = np.where(chunk == eos_id)[0]
    docs = [x.tolist() for x in split_into_docs(chunk, eos_id)]
    doc_bigrams = [set(zip(doc[:-1], doc[1:])) for doc in docs]
    # print([len(doc) for doc in doc_bigrams])
    # print("Merging")
    # if two docs are have similarity, replace the eos with the replace_token_id
    result_indices = []
    sim_scores = []
    for i in range(len(docs) - 1):
        sim = sim_func(doc_bigrams[i], doc_bigrams[i+1])
        sim_scores.append(sim)
        if lower_bound <= sim <= upper_bound:
            result_indices.append(eos_indices[i].item())
    # if len(eos_indices) > 0:
    #     print("EOS indices:", eos_indices, "Skip indices:", result_indices, "Percentage {}/{}={}".format(len(result_indices), len(eos_indices), len(result_indices)/len(eos_indices)))
    #     print("Similarity scores:", sim_scores)
    # else:
    #     print("No EOS indices")
    return result_indices

def merge_neighboring_docs(chunk, sim_func, eos_id, replace_token_id, lower_bound, upper_bound):
    """
    Merge neighboring documents if they have similarity
    Args:
        chunk: an array of tokens
        eos_id: EOS token id used to mark the boundary of documents
        replace_token_id: default is 13, which is the '\n' token

    Returns: a new array with the EOS tokens replaced
    """
    to_replace = get_eos_indices_between_relevant_docs(chunk, sim_func, eos_id, lower_bound, upper_bound)
    # assert that the token at each index is the EOS token
    assert all(chunk[i] == eos_id for i in to_replace), "Not all tokens to replace are EOS tokens"
    chunk[to_replace] = replace_token_id
    return chunk

def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self, filenames, n_chunks, block_size, seed=12345, shuffle=True, wrap=True, num_processes=1, process_rank=0, mask_attn=False, merge_method="none",
            initial_iter=0
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank
        self._mask_attn = mask_attn
        self._merge_method = merge_method
        self._initial_iter = initial_iter

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
            mask_attn=self._mask_attn,
            merge_method=self._merge_method,
            initial_iter=self._initial_iter,
        )


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        self._write_chunk()

def gradual_increase_with_final_hold(curr_iter_num, iters_per_increase, init_mask_length, final_mask_length,
                                     total_cycles):
    cycle_length_up = (final_mask_length - init_mask_length) * iters_per_increase
    cycle_length_down = (final_mask_length - init_mask_length) * iters_per_increase
    cycle_total_length = cycle_length_up + cycle_length_down
    full_cycles = int(total_cycles)
    half_cycle_length = cycle_length_up // 2  # for the final half cycle to reach 8192

    # Determine the current cycle and position within the cycle
    total_cycle_length = full_cycles * cycle_total_length + half_cycle_length
    if curr_iter_num >= total_cycle_length:
        return final_mask_length  # Hold at 8192 after 4.5 cycles

    cycle_count = curr_iter_num // cycle_total_length
    if cycle_count < full_cycles:
        cycle_position = curr_iter_num % cycle_total_length
        if cycle_position < cycle_length_up:
            return min(init_mask_length + (cycle_position // iters_per_increase), final_mask_length)
        return max(final_mask_length - (cycle_position - cycle_length_up) // iters_per_increase, init_mask_length)

    # For the final half cycle, just increase to the final length
    within_half_cycle = curr_iter_num - full_cycles * cycle_total_length
    return min(init_mask_length + within_half_cycle // iters_per_increase, final_mask_length)

def jump_increase_with_final_hold(curr_iter_num, iters_per_increase, init_mask_length, final_mask_length,
                                  total_cycles):
    # Calculate the total number of iterations needed to increase from init_mask_length to final_mask_length in one cycle
    cycle_length_up = (final_mask_length - init_mask_length) * iters_per_increase

    # Determine how many complete cycles have been executed based on the current iteration number
    cycle_count = curr_iter_num // cycle_length_up

    # If we've reached or exceeded the specified number of cycles, hold at final_mask_length permanently
    if cycle_count >= total_cycles:
        return final_mask_length

    # Calculate the position within the current cycle (i.e., where we are in the up-phase of the cycle)
    cycle_position = curr_iter_num % cycle_length_up

    # Calculate the current mask length within this cycle
    # Increase by 1 for every iters_per_increase until reaching final_mask_length
    current_mask_length = init_mask_length + (cycle_position // iters_per_increase)

    # Ensure we don’t exceed final_mask_length within each cycle
    return min(current_mask_length, final_mask_length)
class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap, mask_attn, merge_method, initial_iter):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None
        assert self._block_size-1 in [512, 1024, 2048, 4096, 8192, 16384, 32768, ], "Block size must be one of 512, 1024, 2048, 4096, 8192, 16384, 32768, but got {}".format(self._block_size)
        print('Dataset block size:', self._block_size)
        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0
        print("In iterator, whether we are masking the attention?", mask_attn)
        print("In iterator, the merge method is", merge_method)
        self._mask_attn = mask_attn
        assert self._mask_attn in ["adaptive", "strict", "", "match", "fix2", "fix1", "fix2rerope" , "fix1rerope", "dm1","dm2", "dm4","intradm1","intradm2","intradm4" ,"intradm8",'dm8', "sc1", "sc2","sc3", "sc4", "gd1c4", "jp1c9",
                                   "gd2c2",
                                   "jp2c5",
                                   "gd4c1",
                                   "jp4c3",
                                   "dm1rd", "dm2rd", "dm4rd", "dm8rd", "intradm1rd", "intradm2rd", "intradm4rd", "intradm8rd",
                                   'dmrd1', 'dmrd2', 'dmrd4', 'dmrd8', 'intradmrd1', 'intradmrd2', 'intradmrd4', 'intradmrd8',
                                   'rdall'
                                   'dm1st4', 'dm1st8', 'dm1st16', 'dm1st32', 'dm1st64', 'dm1st128', 'dm1st256', 'dm1st512',
                                    'dm2st4', 'dm2st8', 'dm2st16', 'dm2st32', 'dm2st64', 'dm2st128', 'dm2st256', 'dm2st512',
                                    'dm4st4', 'dm4st8', 'dm4st16', 'dm4st32', 'dm4st64', 'dm4st128', 'dm4st256', 'dm4st512',
                                   ], "Mask attn must be either adaptive or strict, but got {}".format(self._mask_attn)
        self._merge_method = merge_method
        if self._mask_attn == "adaptive":
            assert self._merge_method == "overlap", "Merge method must be overlap when mask_attn is adaptive, but got {}".format(self._merge_method)
        self._load_n_chunks()

        self._iter_num = initial_iter
        self.init_mask_length = 32
        if 'st4' in self._mask_attn:
            self.init_mask_length = 4
        elif 'st8' in self._mask_attn:
            self.init_mask_length = 8
        elif 'st16' in self._mask_attn:
            self.init_mask_length = 16
        elif 'st32' in self._mask_attn:
            self.init_mask_length = 32
        elif 'st64' in self._mask_attn:
            self.init_mask_length = 64
        elif 'st128' in self._mask_attn:
            self.init_mask_length = 128
        elif 'st256' in self._mask_attn:
            self.init_mask_length = 256
        elif 'st512' in self._mask_attn:
            self.init_mask_length = 512

        self.final_mask_length = block_size
        print("In iterator", self._iter_num, "Initial mask length is", self.init_mask_length, "Final mask length is", self.final_mask_length)
        if self._mask_attn == "sc1":
            self.iters_per_increase = 16
            self.middle_mask_length = 1024
            self.end_no_change_point = 16 * 80000  # 80B tokens
        elif self._mask_attn == "sc2":
            self.iters_per_increase = 32
            self.middle_mask_length = 1024
            self.end_no_change_point = 16 * 80000  # 80B tokens
        elif self._mask_attn == "sc3":
            self.iters_per_increase = 16
            self.middle_mask_length = 512
            self.end_no_change_point = 16 * 80000  # 80B tokens
        elif self._mask_attn == "sc4":
            self.iters_per_increase = 32
            self.middle_mask_length = 512
            self.end_no_change_point = 16 * 80000  # 80B tokens
        elif self._mask_attn in ["dm1", "dm2", "dm4", "dm8", "intradm1", "intradm2", "intradm4", "intradm8",
            "dm1rd", "dm2rd", "dm4rd", "dm8rd", "intradm1rd", "intradm2rd", "intradm4rd", "intradm8rd",
            "dmrd1", "dmrd2", "dmrd4", "dmrd8", "intradmrd1", "intradmrd2", "intradmrd4", "intradmrd8",
                                 'dm1st4', 'dm1st8', 'dm1st16', 'dm1st32', 'dm1st64', 'dm1st128', 'dm1st256', 'dm1st512',
                                 'dm2st4', 'dm2st8', 'dm2st16', 'dm2st32', 'dm2st64', 'dm2st128', 'dm2st256', 'dm2st512',
                                 'dm4st4', 'dm4st8', 'dm4st16', 'dm4st32', 'dm4st64', 'dm4st128', 'dm4st256', 'dm4st512',
                                 ]:
            self.iters_per_increase = self.get_iters_per_increase(self._mask_attn)
            assert self.iters_per_increase > 0, "Invalid iters_per_increase for mask_attn {}".format(self._mask_attn)
        elif self._mask_attn == "gd1c4":
            self.iters_per_increase = 16
            self.total_cycles = 4.5
        elif self._mask_attn == "jp1c9":
            self.iters_per_increase = 16
            self.total_cycles = 9
        elif self._mask_attn == "gd2c2":
            self.iters_per_increase = 32
            self.total_cycles = 2.5
        elif self._mask_attn == "jp2c5":
            self.iters_per_increase = 32
            self.total_cycles = 5
        elif self._mask_attn == "gd4c1":
            self.iters_per_increase = 64
            self.total_cycles = 1.5
        elif self._mask_attn == "jp4c3":
            self.iters_per_increase = 64
            self.total_cycles = 3
        else:
            self.iters_per_increase = -1
        self.is_dm_attention = "no"

        if os.getenv("NUM_NODES") is not None:
            num_nodes = int(os.getenv("NUM_NODES"))
            print(f"Adjusting iters_per_increase from {self.iters_per_increase} to {self.iters_per_increase // num_nodes} due to NUM_NODES={num_nodes}")
            self.iters_per_increase = self.iters_per_increase // num_nodes

        if self._mask_attn in ["dm1", "dm2", "dm4", "dm8", "dm1st4", "dm1st8", "dm1st16", "dm2st4", "dm2st8", "dm2st16",
                            "dm4st4", "dm4st8", "dm4st16",
                            "dm1st32", "dm2st32", "dm4st32", "dm8st32", "dm1st64", "dm2st64", "dm4st64", "dm8st64",
                            "dm1st128", "dm2st128", "dm4st128", "dm8st128",
                            "dm1st256", "dm2st256", "dm4st256", "dm8st256", "dm1st512", "dm2st512", "dm4st512",
                            "dm8st512",
                            ]:
            self.is_dm_attention = "dm"
        elif self._mask_attn in ['intradm1', 'intradm2', 'intradm4', 'intradm8']:
            self.is_dm_attention = 'intradm'



    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            if not self._wrap:
                raise StopIteration
            self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                assert self._chunk_size % self._block_size == 0, "Chunk size {} must be a multiple of block size {}".format(self._chunk_size, self._block_size)
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self



    def calculate_mask_length(self, curr_iter_num, iter_per_increase_1):
        curr_mask_length = self.init_mask_length + (curr_iter_num // iter_per_increase_1) * 1
        return min(curr_mask_length, self.final_mask_length)

    def scheduled_mask_length(self, curr_iter_num, iter_per_increase_1, middle_length, end_no_change_point):
        start_no_change_point = (middle_length - self.init_mask_length) * iter_per_increase_1
        # end_no_change_point = 16 * 80000  # 80B tokens

        if curr_iter_num < start_no_change_point:
            # Calculate mask length before the no-change point
            return self.init_mask_length + (curr_iter_num // iter_per_increase_1)
        elif curr_iter_num < end_no_change_point:
            # Fixed mask length during the no-change phase
            return middle_length
        else:
            curr_mask_length = middle_length + (curr_iter_num - end_no_change_point) // iter_per_increase_1
            return min(curr_mask_length, self.final_mask_length)

    def get_iters_per_increase(self, mask_attn):
        iters_per_increase_for_8k = self.get_iters_per_increase_for_8k(mask_attn)
        if self._block_size == 8192 + 1:
            return iters_per_increase_for_8k
        elif self._block_size > 8192 + 1:
            assert (self._block_size - 1) % 8192 == 0, "Block size must be a multiple of 8192, but got {}".format(self.block_size)
            return iters_per_increase_for_8k // ((self._block_size -1 ) // 8192)
        else:
            assert 8192 % (self._block_size - 1) == 0, "Block size must be a fraction of 8192, but got {}".format(self.block_size)
            return iters_per_increase_for_8k * (8192 // (self._block_size-1))

    def get_iters_per_increase_for_8k(self, mask_attn):
        # if mask_attn == "dm1" or mask_attn == "intradm1" or mask_attn == "dm1rd" or mask_attn == "intradm1rd" or mask_attn == "dmrd1" or mask_attn == "intradmrd1":
        if mask_attn in ["dm1", "intradm1", "dm1rd", "intradm1rd", "dmrd1", "intradmrd1", 'dm1st4', 'dm1st8', 'dm1st16', 'dm1st32', 'dm1st64', 'dm1st128', 'dm1st256', 'dm1st512']:
            return 16
        #elif mask_attn == "dm2" or mask_attn == "intradm2" or mask_attn == "dm2rd" or mask_attn == "intradm2rd" or mask_attn == "dmrd2" or mask_attn == "intradmrd2":
        elif mask_attn in ["dm2", "intradm2", "dm2rd", "intradm2rd", "dmrd2", "intradmrd2", 'dm2st4', 'dm2st8', 'dm2st16', 'dm2st32', 'dm2st64', 'dm2st128', 'dm2st256', 'dm2st512']:
            return 32
        #elif mask_attn == "dm4" or mask_attn == "intradm4" or mask_attn == "dm4rd" or mask_attn == "intradm4rd" or mask_attn == "dmrd4" or mask_attn == "intradmrd4":
        elif mask_attn in ["dm4", "intradm4", "dm4rd", "intradm4rd", "dmrd4", "intradmrd4", 'dm4st4', 'dm4st8', 'dm4st16', 'dm4st32', 'dm4st64', 'dm4st128', 'dm4st256', 'dm4st512']:
            return 64
        #elif mask_attn == "dm8" or mask_attn == "intradm8" or mask_attn == "dm8rd" or mask_attn == "intradm8rd" or mask_attn == "dmrd8" or mask_attn == "intradmrd8":
        elif mask_attn in ["dm8", "intradm8", "dm8rd", "intradm8rd", "dmrd8", "intradmrd8"]:
            return 128
        else:
            raise ValueError("Mask attn must be dm1, dm2, or dm4, but got {}".format(mask_attn))

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        arr = torch.from_numpy(arr.astype(np.int64)) # block size here is 8193
        # print("Block size", self._block_size, "arr shape", arr.shape, "arr dtype", arr.dtype, "arr", arr)
        self._curr_idx += 1
        if self._merge_method == "overlap" and self._mask_attn == "strict": # only merge neighboring docs when mask_attn is strict
            arr = merge_neighboring_docs(arr, sim_func=calculate_bigram_set_overlap, eos_id=2, replace_token_id=13, lower_bound=0.1, upper_bound=0.5)
        else:
            assert self._merge_method == "no" or (self._merge_method=="overlap" and self._mask_attn=="adaptive") , "Merge method must be either overlap or no, but got {}".format(self._merge_method)
        if self._mask_attn:
            # if self._mask_attn == "adaptive":
            #     skip_eos_indices = get_eos_indices_between_relevant_docs(arr, sim_func=calculate_bigram_set_overlap, eos_id=2, lower_bound=0.1, upper_bound=0.5)
            #     cur_fragment_lens, cur_fragment_nums = get_fragment_lens_optimized(arr[:self._block_size-1], skip_eos_indices)
            # elif self._mask_attn == "match":
            #     skip_eos_indices = get_random_skip_indices(arr, eos_id=2)
            #     cur_fragment_lens, cur_fragment_nums = get_fragment_lens_optimized(arr[:self._block_size-1], skip_eos_indices)

            # elif self._mask_attn == "sc2":
            #     iters_per_increase = 32
            #     curr_mask_length = self.scheduled_mask_length(self._iter_num, iters_per_increase)
            #     cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
            if self.is_dm_attention == "dm":
                # iters_per_increase = self.get_iters_per_increase(self._mask_attn)
                curr_mask_length = self.calculate_mask_length(self._iter_num, self.iters_per_increase)
                # print("Current iteration number is", self._iter_num, "With masking strategy", self._mask_attn, "Current mask length is", curr_mask_length, "Iters per increase", self.iters_per_increase)
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
            elif self.is_dm_attention == "intradm":
                curr_mask_length = self.calculate_mask_length(self._iter_num, self.iters_per_increase)
                # print("Current iteration number is", self._iter_num, "With masking strategy", self._mask_attn, "Current mask length is", curr_mask_length, "Iters per increase", self.iters_per_increase)
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length_intramask(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
            elif self._mask_attn.startswith("gd"):
                curr_mask_length = gradual_increase_with_final_hold(self._iter_num, self.iters_per_increase, self.init_mask_length, self.final_mask_length, self.total_cycles)
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
                # print("Current iteration number is", self._iter_num, "With masking strategy", self._mask_attn, "Current mask length is", curr_mask_length, "Iters per increase", self.iters_per_increase)
            elif self._mask_attn.startswith("jp"):
                curr_mask_length = jump_increase_with_final_hold(self._iter_num, self.iters_per_increase, self.init_mask_length, self.final_mask_length, self.total_cycles)
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
                #print("Current iteration number is", self._iter_num, "With masking strategy", self._mask_attn, "Current mask length is", curr_mask_length, "Iters per increase", self.iters_per_increase)
            elif self._mask_attn.startswith("rdall"):
                # get a random mask length between init_mask_length and final_mask_length
                curr_mask_length = np.random.randint(self.init_mask_length, self.final_mask_length)
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
            elif self._mask_attn == "fix2" or self._mask_attn == "fix2rerope":
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], 2048)
            elif self._mask_attn == "fix1" or self._mask_attn == "fix1rerope":
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], 1024)
            elif self._mask_attn.startswith("sc"):
                curr_mask_length = self.scheduled_mask_length(self._iter_num, self.iters_per_increase, self.middle_mask_length, self.end_no_change_point)
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
            elif self._mask_attn in ['dm1rd', 'dm2rd', 'dm4rd', 'dm8rd', 'intradm1rd', 'intradm2rd', 'intradm4rd', 'intradm8rd']:
                curr_mask_length = self.calculate_mask_length(self._iter_num, self.iters_per_increase)
                # if the mask length reaches the final mask length, then randomly sample a new mask length
                if curr_mask_length == self.final_mask_length:
                    curr_mask_length = np.random.randint(self.init_mask_length, self.final_mask_length)
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
            elif self._mask_attn in ['dmrd1', 'dmrd2', 'dmrd4', 'dmrd8', 'intradmrd1', 'intradmrd2', 'intradmrd4', 'intradmrd8']:
                curr_mask_length = self.calculate_mask_length(self._iter_num, self.iters_per_increase)
                # add a random noise of +- 100 to the mask length
                curr_mask_length += np.random.randint(-100, 100)
                # ensure that the mask length is within the range
                curr_mask_length = max(self.init_mask_length, min(self.final_mask_length, curr_mask_length))
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size-1], curr_mask_length, is_multiple=False)
            else:
                assert self._mask_attn == "strict", "Mask attn must be either adaptive or strict, but got {}".format(self._mask_attn)
                # cur_fragment_lens, cur_fragment_nums = get_fragment_lens(arr[:self._block_size-1], []) # only calculate the input
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_optimized(arr[:self._block_size-1], [])

                # assert cur_fragment_nums == cur_fragment_nums2, "Fragment nums do not match"
                # assert cur_fragment_lens == cur_fragment_lens2, "Fragment lens do not match"

            # print("cur_fragment_lens", cur_fragment_lens, "cur_fragment_nums", cur_fragment_nums)
            # assert arr[1024] == 2, f"Middle token is not EOS: {arr[1024]}"

            # print("Yieleding with mask attn, shapes are : ", arr.shape, len(cur_fragment_lens), cur_fragment_nums, "Sum of fragment lens", sum(cur_fragment_lens))
            self._iter_num += 1
            return {"idx": arr, "fragment_lens": cur_fragment_lens, "fragment_nums": cur_fragment_nums}
        else:
            # assert arr[1024] == 2, f"Middle token is not EOS: {arr[1024]}"
            return {"idx": arr}


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)
