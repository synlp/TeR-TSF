export CUDA_VISIBLE_DEVICES=4

data_dir=/data2/user2/ter_tsf/qwen3-1.7b/tfhts/weather/reinforced_data/
save_dir=/data2/user2/ter_tsf/TextFusionHTS/
epochs=200


python ./Models/TextFusionHTS/train_tfhts.py \
    --data_dir $data_dir \
    --epochs $epochs \
    --save_dir $save_dir \
    --data_name weather \
    --hist_len 36 \
    --pred_len 6 \
    --batch_size 16 \
    --iter_idx 3 \
    --text_type reinforced_text \
    --exp_time 001 \
    --patch_len 16 \
    --stride 8