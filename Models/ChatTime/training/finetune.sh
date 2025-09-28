DATA_PATH=""
CODE_PATH=""
MODEL_PATH=""

code_path=$CODE_PATH
model_path=$MODEL_PATH/ChatTime-1-7B-Base/
dataset_path=$DATA_PATH/ChatTime-1-Finetune-100K/
log_path=$MODEL_PATH/log_finetune/
output_path=$MODEL_PATH/ChatTime-1-7B-Chat/

lora_rank=8
lora_alpha=16
lora_dropout=0.00

num_train_epochs=4
per_device_train_batch_size=8
gradient_accumulation_steps=32
save_steps=40
logging_steps=4
max_steps=-1

python "$code_path/training/source/finetune.py" \
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
