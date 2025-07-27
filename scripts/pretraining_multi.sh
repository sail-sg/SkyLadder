export WANDB_PROJECT=RetPretrain
export MODEL_NAME=$1
export DATASET_NAME=$2
export VALID_DATASET_NAME=$3
export BINS_ROOT=/home/aiops/zhuty/ret_pretraining_data/sample_processed_qwen

FULL_DATA_PATH=$BINS_ROOT/$DATASET_NAME
VALID_DATA_PATH=$BINS_ROOT/$VALID_DATASET_NAME/valid
export WANDB_API_KEY=5723c6f7e50618fd5dd5a9c2dc2c293f293a25e6

echo "start time: $(date)"

# make sure that the training and validation data folder exist
if [ ! -d $FULL_DATA_PATH ]; then
  echo "Error: $FULL_DATA_PATH does not exist"
  exit 1
fi
if [ ! -d $VALID_DATA_PATH ]; then
  echo "Error: $VALID_DATA_PATH does not exist"
  exit 1
fi
export resume=$4
export eval_only=$5
export suffix=$6

# check if we need to resume
if [[ $resume == "true" ]]; then
  echo "Resuming from checkpoint"
  export resume=true
else
  echo "Not resuming from checkpoint"
  export resume=false
fi

if [[ $eval_only == "true" ]]; then
  echo "Evaluating only"
  export eval_only=true
else
  echo "Training and evaluating"
  export eval_only=false
fi

export NUM_NODES=4
# if [[ $MODEL_NAME != tiny_LLaMA_* ]]; then
#   echo "Here, Error: '$MODEL_NAME' is not a valid model name."
#   exit 1
# fi


export GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
export WANDB_NAME=$MODEL_NAME\_$DATASET_NAME
if [[ $suffix != "" ]]; then
  echo "Adding suffix $suffix"
  export WANDB_NAME=$WANDB_NAME\_$suffix
  echo "WANDB_NAME=$WANDB_NAME"
fi
export NUMBER_OF_GPU=$(python -c "import torch; print(torch.cuda.device_count())")
export WANDB_TAGS="pretraining,$DATASET_NAME,$MODEL_NAME"
echo "Using $NUMBER_OF_GPU GPUs"
echo "WANDB_NAME=$WANDB_NAME"
echo "WANDB_TAGS=$WANDB_TAGS"

lightning run model \
    --node-rank=$RANK  \
    --main-address=$MASTER_ADDR \
    --main-port=$MASTER_PORT \
    --accelerator=cuda \
    --num-nodes=$NUM_NODES \
    --devices=$NUMBER_OF_GPU \
    pretrain/tinyllama.py --num_devices $NUMBER_OF_GPU \
    --train_data_dir $FULL_DATA_PATH \
    --val_data_dir $VALID_DATA_PATH \
    --resume $resume \
    --eval_only $eval_only

# sample usage
# bash scripts/pretraining.sh redpajama_2b
# bash scripts/pretraining.sh redpajama_2b_reordered_train_top10