# export CUDA_VISIBLE_DEVICES=0
bash ./main.sh \
    --exp_time 010 \
    --llm_type qwen3-1.7b \
    --tsf_type tfhts \
    --data_name Health_US \
    --hist_len 96 \
    --pred_len 48 \
    --batch_size 64 \
    --gen_num 2 \
    --iter_num 5 \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --lr 5.0e-5 \
    --num_train_epochs 5 \
    --llama_factory_dir "llama-factory-main" \
    --disable_text_quality_reward \
    --down_sample 0

# bash ./main.sh \
#     --exp_time 001 \
#     --llm_type qwen3-4b \
#     --tsf_type tfhts \
#     --data_name Energy \
#     --hist_len 96 \
#     --pred_len 12 \
#     --batch_size 32 \
#     --gen_num 2 \
#     --iter_num 5 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 1 \
#     --lr 5.0e-5 \
#     --num_train_epochs 5 \
#     --llama_factory_dir "llama-factory-main" \
#     --down_sample 0