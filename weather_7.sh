# export CUDA_VISIBLE_DEVICES=4
bash ./main.sh \
    --exp_time 002 \
    --llm_type qwen3-1.7b \
    --tsf_type tfhts \
    --data_name weather \
    --hist_len 14 \
    --pred_len 7 \
    --batch_size 64 \
    --gen_num 2 \
    --iter_num 5 \
    --lora_rank 16 \
    --per_device_train_batch_size 2 \
    --lr 5.0e-5 \
    --num_train_epochs 8 \
    --llama_factory_dir "llama-factory-main" \
    --down_sample 0

