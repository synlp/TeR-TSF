export CUDA_VISIBLE_DEVICES=1

data_dir=/data2/user2/ter_tsf/processed_data/
save_dir=/data2/user2/ter_tsf/TextFusionHTS/
epochs=200

python ./Models/TextFusionHTS/train_tfhts.py \
    --data_dir $data_dir \
    --epochs $epochs \
    --save_dir $save_dir \
    --data_name ETTh1 \
    --hist_len 336 \
    --pred_len 96 \
    --batch_size 64 \
    --text_type original_text \
    --patience 10 \
    --patch_len 16 \
    --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name MTBench_finance \
#     --hist_len 96 \
#     --pred_len 24 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name MTBench_weather \
#     --hist_len 96 \
#     --pred_len 12 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name weather \
#     --hist_len 14 \
#     --pred_len 3 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 4 \
#     --stride 2

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name weather \
#     --hist_len 14 \
#     --pred_len 7 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 4 \
#     --stride 2

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name weather \
#     --hist_len 14 \
#     --pred_len 14 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 4 \
#     --stride 2


# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Heart_Rate \
#     --hist_len 36 \
#     --pred_len 6 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Heart_Rate \
#     --hist_len 36 \
#     --pred_len 12 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Heart_Rate \
#     --hist_len 36 \
#     --pred_len 18 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Energy \
#     --hist_len 96 \
#     --pred_len 12 \
#     --batch_size 16 \
#     --iter_idx 0 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Energy \
#     --hist_len 96 \
#     --pred_len 24 \
#     --batch_size 16 \
#     --iter_idx 0 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Energy \
#     --hist_len 96 \
#     --pred_len 48 \
#     --batch_size 16 \
#     --iter_idx 0 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Agriculture \
#     --hist_len 36 \
#     --pred_len 6 \
#     --batch_size 16 \
#     --iter_idx 0 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 


# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Agriculture \
#     --hist_len 36 \
#     --pred_len 12 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Agriculture \
#     --hist_len 36 \
#     --pred_len 18 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Climate \
#     --hist_len 96 \
#     --pred_len 12 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Climate \
#     --hist_len 96 \
#     --pred_len 24 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Climate \
#     --hist_len 96 \
#     --pred_len 48 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Economy \
#     --hist_len 36 \
#     --pred_len 6 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Economy \
#     --hist_len 36 \
#     --pred_len 12 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Economy \
#     --hist_len 36 \
#     --pred_len 18 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Environment \
#     --hist_len 336 \
#     --pred_len 48 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Environment \
#     --hist_len 336 \
#     --pred_len 96 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Environment \
#     --hist_len 336 \
#     --pred_len 192 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Health_US \
#     --hist_len 96 \
#     --pred_len 12 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Health_US \
#     --hist_len 96 \
#     --pred_len 24 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Health_US \
#     --hist_len 96 \
#     --pred_len 48 \
#     --batch_size 64 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name SocialGood \
#     --hist_len 36 \
#     --pred_len 6 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name SocialGood \
#     --hist_len 36 \
#     --pred_len 12 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name SocialGood \
#     --hist_len 36 \
#     --pred_len 18 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Traffic \
#     --hist_len 36 \
#     --pred_len 6 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Traffic \
#     --hist_len 36 \
#     --pred_len 12 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name Traffic \
#     --hist_len 36 \
#     --pred_len 18 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir /data2/user2/ter_tsf/processed_data/ \
#     --epochs 200 \
#     --save_dir /data2/user2/ter_tsf/TextFusionHTS/ \
#     --data_name weather \
#     --hist_len 36 \
#     --pred_len 6 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 8 \
#     --stride 4 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name weather \
#     --hist_len 36 \
#     --pred_len 12 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 8 \
#     --stride 4 \
#     --use_text 0 

# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name weather \
#     --hist_len 36 \
#     --pred_len 18 \
#     --batch_size 16 \
#     --text_type original_text \
#     --patch_len 8 \
#     --stride 4 \
#     --use_text 0 


# python ./Models/TextFusionHTS/train_tfhts_4vis.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name exchange_rate \
#     --hist_len 336 \
#     --pred_len 96 \
#     --batch_size 32 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name exchange_rate \
#     --hist_len 336 \
#     --pred_len 192 \
#     --batch_size 32 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name exchange_rate \
#     --hist_len 336 \
#     --pred_len 336 \
#     --batch_size 32 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name ETTh1 \
#     --hist_len 336 \
#     --pred_len 96 \
#     --batch_size 32 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name ETTh1 \
#     --hist_len 336 \
#     --pred_len 192 \
#     --batch_size 32 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8

# python ./Models/TextFusionHTS/train_tfhts.py \
#     --data_dir $data_dir \
#     --epochs $epochs \
#     --save_dir $save_dir \
#     --data_name ETTh1 \
#     --hist_len 336 \
#     --pred_len 336 \
#     --batch_size 32 \
#     --text_type original_text \
#     --patch_len 16 \
#     --stride 8