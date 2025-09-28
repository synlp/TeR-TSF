DATA_PATH=""
CODE_PATH=""
MODEL_PATH=""

code_path=$CODE_PATH
model_path=meta-llama/Llama-2-7b-hf
dataset_path=$DATA_PATH/ChatTime-1-Pretrain-1M/
log_path=$MODEL_PATH/log_pretrain/
output_path=$MODEL_PATH/ChatTime-1-7B-Base/

lora_rank=8
lora_alpha=16
lora_dropout=0.00

num_train_epochs=2
per_device_train_batch_size=8
gradient_accumulation_steps=32
save_steps=200
logging_steps=20
max_steps=-1

python "$code_path/training/source/pretrain.py" \
  --code_path "$code_path" \
  --model_path "$model_path" \
  --dataset_path "$dataset_path" \
  --log_path "$log_path" \
  --output_path "$output_path" \
  --lora_rank $lora_rank \
  --lora_alpha $lora_alpha \
  --lora_dropout $lora_dropout \
  --num_train_epochs $num_train_epochs \
  --per_device_train_batch_size $per_device_train_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --save_steps $save_steps \
  --logging_steps $logging_steps \
  --max_steps $max_steps \
  --load_in_4bit
