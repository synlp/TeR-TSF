export CUDA_VISIBLE_DEVICES=2

python Models/Time_LLM/train_custom.py \
    --data_path /data2/user2/rl_tsf/All_data/processed_data/weather_5_5_train.csv \
    --data weather \
    --seq_len 5 \
    --pred_len 5 \
    --llm_model LLAMA \
    --batch_size 64 \
    --train_epochs 100 \
    --patch_len 3 \
    --stride 2 \
    --patience 3

python Models/Time_LLM/train_custom.py \
    --data_path /data2/user2/rl_tsf/All_data/processed_data/weather_6_6_train.csv \
    --data weather \
    --seq_len 6 \
    --pred_len 6 \
    --llm_model LLAMA \
    --batch_size 64 \
    --train_epochs 100 \
    --patch_len 3 \
    --stride 2 \
    --patience 3

python Models/Time_LLM/train_custom.py \
    --data_path /data2/user2/rl_tsf/All_data/processed_data/weather_7_7_train.csv \
    --data weather \
    --seq_len 7 \
    --pred_len 7 \
    --llm_model LLAMA \
    --batch_size 64 \
    --train_epochs 100 \
    --patch_len 3 \
    --stride 2 \
    --patience 3

# MODEL_PATH=$(find ./checkpoints -name "custom_model*" -type d | head -1)

# python inference_custom.py \
#             --data_path Agriculture_36_6_train.csv \
#             --model_path $MODEL_PATH/checkpoint.pth \
#             --seq_len $SEQ_LEN \
#             --pred_len $PRED_LEN \
#             --llm_model $LLM_MODEL \
#             --batch_size $BATCH_SIZE