export CUDA_VISIBLE_DEVICES=1

data_dir=/data2/user2/ter_tsf/processed_data/
save_dir=/data2/user2/ter_tsf/TextFusionHTS/
epochs=200

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name exchange_rate \
#     --hist_len 336 \
#     --pred_len 336 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

python ./Models/TextFusionHTS/train_tfhts_4vis.py \
    --data_dir $data_dir \
    --epochs $epochs \
    --save_dir $save_dir \
    --data_name ETTh1 \
    --hist_len 336 \
    --pred_len 336 \
    --batch_size 64 \
    --text_type original_text \
    --patch_len 16 \
    --stride 8 \
    --use_text 0 