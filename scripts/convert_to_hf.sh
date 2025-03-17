
LIT_CKPT_DIR=$1
CKPT_FILENAME=$2
CKPT_PREFIX=$(basename "$CKPT_FILENAME" .pth)

patterns=(
  "tiny_LLaMA_([0-9a-zA-Z]+)_([0-9]+k)(_intramask)?"
  "llama3.2_([0-9a-zA-Z]+)_([0-9]+k)(_intramask)?"
  "tiny_LLaMA_([0-9]+[M|b])_([0-9]+)(_intramask)?"
)

# Flag to track if a match is found
match_found=false

# Loop through the patterns and try to match
for pattern in "${patterns[@]}"; do
  if [[ $LIT_CKPT_DIR =~ $pattern ]]; then
    MODEL_NAME="${BASH_REMATCH[0]}"
    echo "MODEL_NAME $MODEL_NAME"
    match_found=true
    break  # Exit the loop once a match is found
  fi
done

# If no pattern matches, handle the "model not found" case
if [[ $match_found == false ]]; then
  echo "MODEL_NAME not found"
  exit 1
fi



OUTPUT_DIR=$LIT_CKPT_DIR/$CKPT_PREFIX\_hf
mkdir -p $OUTPUT_DIR
echo "OUTPUT_DIR $OUTPUT_DIR"
python scripts/convert_lit_checkpoint.py \
--checkpoint_name $CKPT_PREFIX.pth \
--out_dir $LIT_CKPT_DIR \
--model_name $MODEL_NAME \
--model_only false


# move config
mv $LIT_CKPT_DIR/config.json $OUTPUT_DIR/config.json
mv $LIT_CKPT_DIR/$CKPT_PREFIX.bin $OUTPUT_DIR/pytorch_model.bin
