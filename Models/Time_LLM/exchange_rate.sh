export CUDA_VISIBLE_DEVICES=1

python Models/Time_LLM/train_custom.py \
    --data_path /data2/user2/rl_tsf/All_data/processed_data/exchange_rate_336_96_train.csv \
    --data exchange_rate \
    --seq_len 336 \
    --pred_len 96 \
    --llm_model LLAMA \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 3

python Models/Time_LLM/train_custom.py \
    --data_path /data2/user2/rl_tsf/All_data/processed_data/exchange_rate_336_192_train.csv \
    --data exchange_rate \
    --seq_len 336 \
    --pred_len 192 \
    --llm_model LLAMA \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 3

python Models/Time_LLM/train_custom.py \
    --data_path /data2/user2/rl_tsf/All_data/processed_data/exchange_rate_336_336_train.csv \
    --data exchange_rate \
    --seq_len 336 \
    --pred_len 336 \
    --llm_model LLAMA \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 3

# MODEL_PATH=$(find ./checkpoints -name "custom_model*" -type d | head -1)

# python inference_custom.py \
#             --data_path Agriculture_36_6_train.csv \
#             --model_path $MODEL_PATH/checkpoint.pth \
#             --seq_len $SEQ_LEN \
#             --pred_len $PRED_LEN \
#             --llm_model $LLM_MODEL \
#             --batch_size $BATCH_SIZE