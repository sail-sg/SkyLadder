export WANDB_PROJECT=RetPretrain
export DATASET_NAME=$2
export VALID_DATASET_NAME=$3
FULL_DATA_PATH=$BINS_ROOT/$DATASET_NAME
VALID_DATA_PATH=$BINS_ROOT/$VALID_DATASET_NAME/valid
export MODEL_NAME=$1
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

if [[ $MODEL_NAME != tiny_LLaMA_* ]]; then
  echo "Here, Error: '$MODEL_NAME' is not a valid model name."
  exit 1
fi

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
    --node-rank=0  \
    --main-address=127.0.01 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPU \
    pretrain/tinyllama.py --num_devices $NUMBER_OF_GPU \
    --train_data_dir $FULL_DATA_PATH \
    --val_data_dir $VALID_DATA_PATH \
    --resume $resume \
    --eval_only $eval_only

# sample usage
# bash scripts/pretraining.sh redpajama_2b
# bash scripts/pretraining.sh redpajama_2b_reordered_train_top10