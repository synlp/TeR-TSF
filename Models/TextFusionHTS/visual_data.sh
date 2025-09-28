export CUDA_VISIBLE_DEVICES=0

save_dir=/data2/user2/ter_tsf/visual/
data_name=SocialGood
hist_len=36
pred_len=12
epochs=200

python ./Models/TextFusionHTS/train_tfhts_4vis.py \
    --data_dir /data2/user2/ter_tsf/processed_data/ \
    --epochs $epochs \
    --save_dir $save_dir \
    --data_name $data_name \
    --hist_len $hist_len \
    --pred_len $pred_len \
    --batch_size 16 \
    --iter_idx 0 \
    --text_type original_text \
    --patch_len 16 \
    --stride 8 \
    --use_text 0 \
    --save_output

python ./Models/TextFusionHTS/train_tfhts_4vis.py \
    --data_dir /data2/user2/ter_tsf/processed_data/ \
    --epochs $epochs \
    --save_dir $save_dir \
    --data_name $data_name \
    --hist_len $hist_len \
    --pred_len $pred_len \
    --batch_size 16 \
    --iter_idx 0 \
    --text_type original_text \
    --patch_len 16 \
    --stride 8 \
    --use_text 1 \
    --save_output

python ./Models/TextFusionHTS/train_tfhts_4vis.py \
    --data_dir /data2/user2/ter_tsf/qwen3-1.7b/tfhts/${data_name}/reinforced_data/ \
    --epochs $epochs \
    --save_dir $save_dir \
    --data_name $data_name \
    --hist_len $hist_len \
    --pred_len $pred_len \
    --batch_size 16 \
    --iter_idx 3 \
    --exp_time 010 \
    --text_type reinforced_text \
    --patch_len 16 \
    --stride 8 \
    --use_text 1 \
    --save_output