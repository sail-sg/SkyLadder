# <img src="assets/ladder.png" alt="ladder" style="width:1em; vertical-align:middle;">SkyLadder: Better and Faster Pretraining via Context Window Scheduling

[![Paper](https://img.shields.io/badge/Read-Paper-blue?style=for-the-badge&logo=googlescholar)](https://arxiv.org/abs/2503.15450)
[![Slides](https://img.shields.io/badge/View-Slides-green?style=for-the-badge&logo=googlemeet)](https://docs.google.com/presentation/d/1Uot7hQRRBAxFsppVDUpMEMpw_H6bcruNIf3eMmdFqeY/edit?usp=sharing)


## Introduction
Recent advancements in LLM pretraining have featured ever-expanding context
windows to process longer sequences. However, our pilot study reveals that models
pretrained with shorter context windows consistently outperform their long-context
counterparts under a fixed token budget. This finding motivates us to explore
an optimal **context window scheduling** strategy to better balance long-context
capability with pretraining efficiency. To this end, we propose SkyLadder, a simple
yet effective approach that implements a short-to-long context window transition.
SkyLadder preserves strong standard benchmark performance, while matching or
exceeding baseline results on long-context tasks. Through extensive experiments,
we pre-train 1B-parameter models (up to 32K context) and 3B-parameter models
(8K context) on 100B tokens, demonstrating that SkyLadder yields consistent gains
of up to 3.7% on common benchmarks, while achieving up to 22% faster training
speeds compared to baselines

<p align="center">
  <img src="assets/1b-8k-proweb.png" alt="training loss" width="50%">
  <img src="assets/schedule_illustration_v2.png" alt="illustration" width="45%">
</p>

## Quick Start 

This project is based on the [TinyLlama](https://github.com/jzhang38/TinyLlama) project. It has been adapted to support pretraining with context window scheduling, intra-document masking, etc.

###  Installation
If you already an environment built for [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md), you can directly use it. 
Otherwise, please use the following commands to build a new environment. 
Here, we expect a CUDA version of 11.8
```bash
conda create -n ladder-pretrain python=3.8
conda activate ladder-pretrain
# install the latest compatible version of torch and xformers, this should install torch 2.4.1
pip install ninja
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118 
# install flash attention
git clone --branch v2.3.3 --depth 1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention
# install other dependencies 
pip install -r requirements.txt
```
If you wish to use Docker, you could follow the instruction [here](https://github.com/sail-sg/regmix/tree/main/model_training) for the docker image name `siviltaramqian/tinyllama:latest`

### Data preparation
The data preparation process is the same as the original tinyllama project.
First make sure that your data is in jsonl format in one directory of the following structure:
```
TEXT_DIR
├── cc
│   ├── train
│   │   ├── 0.jsonl
│   │   ├── 1.jsonl
│   │   └── ...
│   └── valid
│       ├── 0.jsonl
│       ├── 1.jsonl
│       └── ...
└── ...
```
You can download the 30B corpus of [CommonCrawl](https://huggingface.co/datasets/tyzhu/cc_subset) (from SlimPajama), and the high-quality [FineWeb-pro](https://huggingface.co/datasets/gair-prox/FineWeb-pro) dataset.

Then run the following:

```bash
export TEXT_DIR=<YOUR_TEXT_DIR>
export BINS_ROOT=<YOUR_BIN_DIR> # where to store the processed chunks
bash scripts/pajama_processing.sh cc 8k
```
where `cc` is the dataset name and `8k` is the sequence length (supporting from 512 to 16k). 
The `TEXT_DIR` is the directory where the text data is stored and the `BIN_DIR` is the directory where the processed data will be stored.
After this step, you will have the data in the following structure:
```
BINS_ROOT
├── cc_8k
│   ├── train_0.bin
│   ├── train_1.bin
│   ├── ...
│   ├── valid_0.bin
│   ├── valid_1.bin
│   └── ...
└── ...
```

### Pretraining
Next, you can start pretraining by running the following:
```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY> # if you want to log into wandb
export BINS_ROOT=<YOUR_BIN_DIR> # from the previous data preparation step
bash scripts/pretraining.sh tiny_LLaMA_1b_8k cc_8k cc_8k # replace 1b_8k with 120M_8k or 360M_8k for smaller models
```
The general usage of `pretraining.sh` is `bash scripts/pretraining.sh model_config_name train_dataset_name eval_dataset_name`. For instance, 
`tiny_LLaMA_1b_8k` is the model config name, `cc_8k` is the training dataset name, and `cc_8k` is the evaluation dataset name. 
The script will look for bins created in the previous step. Those with a `train_*` prefix are used for training and those with a `valid_*` prefix are used for evaluation.

You can simply replace the model config name to get different models:
```
bash scripts/pretraining.sh tiny_LLaMA_1b_8k cc_8k cc_8k # baseline with standard causal attention
bash scripts/pretraining.sh tiny_LLaMA_1b_8k_intramask cc_8k cc_8k # intradocument masking
bash scripts/pretraining.sh tiny_LLaMA_1b_8k_dm8 cc_8k cc_8k # skyladder with alpha=1/8 
bash scripts/pretraining.sh tiny_LLaMA_1b_8k_intradm8 cc_8k cc_8k # intradocument masking + skyladder with alpha=1/8
```
Here, `dm8` means that $\alpha$ is 1/8: the local window size $w$ will increase by 1 every 8 steps. Therefore, it takes 64k steps to reach 8k. In our implementation, `dm1` is the fastest (8k steps to reach 8k) and `dm8` is the slowest (64k steps to reach 8k).

On a node with 8 A100 (40G) GPUs, the pretraining of a 1B model with 8k context, 100B token takes around 10 days. 
If you wish to get the results faster, do consider using a smaller model. For instance, the 120M (`tiny_LLaMA_120M_8k`) model takes around 1 day to pretrain with 100B tokens. 
Additional pretraining setups (learning rate, batch size, max steps, etc.) should be changed in the [`pretrain/tinyllama.py`](pretrain/tinyllama.py) file.

### Advanced Usage

#### Multi-node Pretraining
Alternatively, if you are running on multiple nodes, you can use the following command:
```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY> # if you want to log into wandb
export BINS_ROOT=<YOUR_BIN_DIR> # from the previous data preparation step
export NUM_NODES=4 # number of nodes, adjust accordingly
bash scripts/pretraining_multi.sh tiny_LLaMA_1b_8k cc_8k cc_8k
```
This will run the pretraining on multiple nodes. 

#### Intra-Document Masking
We implemented intra-document masking (which can be combined with SkyLadder). 
The model name of `tiny_LLaMA_1b_8k_intramask` means that the model will be trained with intra-document masking. 
To combine with SkyLadder, use suffices like `intradm8` ($\alpha=1/8$), `intradm4`, etc.

#### Other types of schedules 
You can also find other types of schedules we experimented with in our paper.
For instance, `tiny_LLaMA_1b_8k_sin8` means that the schedule is a sinusoidal schedule with $\alpha$ being 1/8. 
We support linear (`dm8`), sinusoidal (`sin8`), and exponential (`exp8`) schedules. 
There are two modes we support, based on (1) the rate of increasing the context window or (2) the percentage of training tokens with an increasing context window. 
1. `{schedule-type}{rate}` where `rate` is $1/\alpha$. For instance, `sin8` means that the context window will increase by 1 every 8 steps.
2. `{schedule-type}{scheduling-percent}p` where `scheduling-percent` is the percentage of training tokens with an increasing context window, "climbing the ladder". For instance, `sin70p` means that 70% of the training tokens will have an increasing context window, following a sinusoidal schedule.



#### Convert to HF
You can convert the model to Huggingface format by running the following:
```bash
bash scripts/convert_to_hf.sh <SAVE_DIR_OF_MODEL> <CHECKPOINT_FILENAME>
# Sample usage
bash scripts/convert_to_hf.sh out/tiny_LLaMA_120M_8k_exp8_cc_8k iter-120000-ckpt-step-30000.pth
```
where `SAVE_DIR_OF_MODEL` is the directory where the litgpt model is saved and `CHECKPOINT_FILENAME` is the checkpoint file to be converted.
The converted model will be saved in the `SAVE_DIR_OF_MODEL` directory.


## Key Changes
The following content describes the key changes we made to the original TinyLlama project.
It is helpful if you would like to adopt our changes to other pretraining libraries.
There are mainly two parts that are changed: 
1. Data loading: Along with the original token ids, the dataset iterator would also output the fragment lengths and the number of fragments. Attention is calculated within each fragment, and not across fragments. 
2. Attention calculation in model: The model would use the fragment lengths to calculate the attention mask, using the `flash_attn_varlen_func` function.

#### Calculating fragment lengths
We use the following snippets to calculate the fragment lengths in the [`lit_gpt/packed_dataset.py`](lit_gpt/packed_dataset.py) file.
```python
def get_fragment_lens_fixed_length(chunk, fixed_length, is_multiple=True):
    assert fixed_length > 0, "Fixed length must be greater than 0, but got {}".format(fixed_length)
    filtered_indices = np.arange(fixed_length, len(chunk),
                                 fixed_length) - 1 # Subtract 1 to get the last index of each fragment
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
```
This is called by the dataset iterator when yielding a data point:
```python
if self.is_dm_attention == "dm":
    curr_mask_length = self.calculate_mask_length_with_rounding(self._iter_num, round_to=self.round_to)
    cur_fragment_lens, cur_fragment_nums = get_fragment_lens_fixed_length(arr[:self._block_size - 1],
                                                                          curr_mask_length,
                                                                          is_multiple=False)
return {"idx": arr, "fragment_lens": cur_fragment_lens, "fragment_nums": cur_fragment_nums}
```
The `calculate_mask_length_with_rounding` function is the scheduling function.
For instance, the linear schedule is implemented as follows:
```python
def calculate_mask_length_linear_schedule(self, curr_iter_num):
    if curr_iter_num >= self.changing_point:
        return self.final_mask_length
    else:
        curr_mask_length = self.init_mask_length + int(
            (self.final_mask_length - self.init_mask_length) * (curr_iter_num / self.changing_point))
        return curr_mask_length
```
The intra-document masking is implemented similarly, and the main change is that the fragment lengths are calculated based on the documents lengths, rather than the fixed length.

#### Attention masking
We use the following snippets enable masked attention in the [`lit_gpt/model.py`](lit_gpt/model.py) file.

```diff
- from flash_attn import flash_attn_func
- result = flash_attn_func(
-     q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True
- )

+ from flash_attn import flash_attn_varlen_func
+ q = q.reshape(-1, q.shape[-2], q.shape[-1])
+ k = k.reshape(-1, k.shape[-2], k.shape[-1])
+ v = v.reshape(-1, v.shape[-2], v.shape[-1])
+ 
+ result = flash_attn_varlen_func(
+     q, k, v,
+     cu_seqlens_q=cuseq_lens,
+     cu_seqlens_k=cuseq_lens,
+     max_seqlen_q=max_seqlen,
+     max_seqlen_k=max_seqlen,
+     dropout_p=0.0,
+     softmax_scale=scale,
+     causal=True
+ )
+ 
+ result = result.reshape(bsize, seqlen, nhead, head_dim)
```


## Citation
If you find our paper or this code useful, we would appreciate it if you could cite our work: 
```
@misc{zhu2025skyladderbetterfasterpretraining,
      title={SkyLadder: Better and Faster Pretraining via Context Window Scheduling}, 
      author={Tongyao Zhu and Qian Liu and Haonan Wang and Shiqi Chen and Xiangming Gu and Tianyu Pang and Min-Yen Kan},
      year={2025},
      eprint={2503.15450},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.15450}, 
}
```

## Acknowledgement
We would like to thank the authors of the original [TinyLLaMA](https://github.com/jzhang38/TinyLlama) project for their work, so that we can build on top of it.
The ladder icon in the title is made by [Freepik](https://www.flaticon.com/authors/freepik) from [www.flaticon.com](https://www.flaticon.com/).. 
