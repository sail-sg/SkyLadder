
This is an extension of tinyllama project that has been adapted for pretraining usage.
We support intra-document masking, local window masking, etc. 

#### Data preparation
The data preparation process is the same as the original tinyllama project.
First make should that your data is in jsonl format in one directory. The run the following:
```bash
export TEXT_DIR=YOUR_TEXT_DIR
export BIN_DIR=OUTPUT_DIR
bash scripts/pajama_processing.sh cc 8k
```
where `cc` is the dataset name and `8k` is the number of tokens in the vocabulary. 
The `TEXT_DIR` is the directory where the text data is stored and the `BIN_DIR` is the directory where the processed data will be stored.
In other words, your training text jsonl files should be in `$TEXT_DIR/cc/train` and the evaluation files should be in `$TEXT_DIR/cc/valid`, and the processed data will be stored in `$BIN_DIR/cc`

#### Pretraining
Next, you can start pretraining by running the following:
```bash
export WANDB_API_KEY=xxxxxxx
export BINS_ROOT=$BIN_DIR # replace with your own
bash scripts/pretraining.sh tiny_LLaMA_1b_8k cc_8k cc_8k
```
First, you need to set the `WANDB_API_KEY` to your own key. The `BINS_DIR` is the directory where the processed data is stored.
The usage of `pretraining.sh` is 'bash scripts/pretraining.sh model_config_name train_dataset_name eval_dataset_name'. For instance, 
`tiny_LLaMA_1b_8k` is the model config name, `cc_8k` is the training dataset name, and `cc_8k` is the evaluation dataset name. 
Please make sure that there is a folder named $BIN_DIR/$val_dataset_name/valid in the `BINS_DIR` directory.


You can simply replace the model config name to get different models:
```
bash scripts/pretraining.sh tiny_LLaMA_1b_8k cc_8k cc_8k
bash scripts/pretraining.sh tiny_LLaMA_1b_8k_intramask cc_8k cc_8k # intradocument masking
bash scripts/pretraining.sh tiny_LLaMA_1b_8k_dm8 cc_8k cc_8k # short-to-long dynamic increase 
bash scripts/pretraining.sh tiny_LLaMA_1b_8k_intradm8 cc_8k cc_8k # intradocument masking + short-to-long dynamic increase
```
Here, dm8 means that $\alpha$ is 1/8. Therefore dm1 is the fastest and dm8 is the slowest.

Alternatively, if you are running on multiple nodes, you can use the following command:
```bash
export WANDB_API_KEY=xxxxxxx
export BINS_ROOT=XXXX # replace with your own
export NUM_NODES=4 # number of nodes, adjust accordingly
bash scripts/pretraining_multi.sh tiny_LLaMA_1b_8k cc_8k cc_8k
```