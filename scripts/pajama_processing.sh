# example dataset name: redpajama_2b
DATASET_NAME=$1
PROCESSING_LENGTH=$2
export SOURCE_PATH=$TEXT_DIR/$DATASET_NAME
export DEST_PATH=$BIN_DIR/$DATASET_NAME

# if charpercent is not empty, add it to the destination path

export TK_PATH=./tokenizer/

if [ $PROCESSING_LENGTH == "4k" ]; then
    chunk_size=65552 # (4096 + 1) * 16
    DEST_PATH=$DEST_PATH\_4k
elif [ $PROCESSING_LENGTH == "256" ]; then
    chunk_size=4112 # (256 + 1)*16
    DEST_PATH=$DEST_PATH\_256
  elif [ $PROCESSING_LENGTH == "128" ]; then
    chunk_size=2064 # (128 + 1)*16
    DEST_PATH=$DEST_PATH\_128
  elif [ $PROCESSING_LENGTH == "64" ]; then
    chunk_size=4160 # (64 + 1)*16 * 4 # need to be larger
    DEST_PATH=$DEST_PATH\_64
elif [ $PROCESSING_LENGTH == "512" ]; then
    chunk_size=8208 # (512 + 1) * 16
    DEST_PATH=$DEST_PATH\_512
elif [ $PROCESSING_LENGTH == "2k" ]; then
    # chunk_size=2098176 # (2048 + 1) * 1024 # changed to 2M for temporary fix for southeast asian languages and repeat setup
    chunk_size=32784 # (2048 + 1) * 16
    DEST_PATH=$DEST_PATH\_2k
elif [ $PROCESSING_LENGTH == "1k" ]; then
    chunk_size=16400 # (1024 + 1) * 16
    # chunk_size=1049600 # (1024 + 1) * 1024 # changed to 1M for temporary fix for repeat setup
    DEST_PATH=$DEST_PATH\_1k
elif [ $PROCESSING_LENGTH == "8k" ]; then
    chunk_size=131088 # (8192 + 1) * 16
    DEST_PATH=$DEST_PATH\_8k
elif [ $PROCESSING_LENGTH == "16k" ]; then
    chunk_size=262160 # (16384 + 1) * 16
    DEST_PATH=$DEST_PATH\_16k
elif [ $PROCESSING_LENGTH == "32k" ]; then
    # chunk_size=33555456 # (32768 + 1) * 1024
    chunk_size=524304 # (32768 + 1) * 16
    DEST_PATH=$DEST_PATH\_32k
else
   chunk_size=32784 # (2048 + 1) * 16
fi
# DEST_PATH=$DEST_PATH\_trunc

echo "Destination path: $DEST_PATH"
echo "Chunk size: $chunk_size"


for split in 'train' 'valid' ; do
# python scripts/prepare_file.py --source_path $SOURCE_PATH \
# python scripts/prepare_repeat_file.py --source_path $SOURCE_PATH \
python scripts/prepare_file.py --source_path $SOURCE_PATH \
--chunk_size $chunk_size --tokenizer_path $TK_PATH --destination_path $DEST_PATH  --short_name $DATASET_NAME --split $split  --text_key text
done
