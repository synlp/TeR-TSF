# export CUDA_VISIBLE_DEVICES=1
bash ./main.sh \
    --exp_time 001 \
    --llm_type qwen3-1.7b \
    --tsf_type tfhts \
    --data_name exchange_rate \
    --hist_len 336 \
    --pred_len 192 \
    --batch_size 32 \
    --gen_num 2 \
    --iter_num 5 \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --lr 5.0e-5 \
    --num_train_epochs 5 \
    --llama_factory_dir "llama-factory-main" \
    --down_sample 0

