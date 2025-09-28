export CUDA_VISIBLE_DEVICES=7

python ./Models/TextFusionHTS/train_tfhts.py --data_path /data2/user2/rl_tsf/All_data/processed_data/exchange_rate_336_96_train.csv \
    --epochs 100 --batch_size 16 --seq_len 336 --pred_len 96 --text_type original_text --data_name exchange_rate
python ./Models/TextFusionHTS/train_tfhts.py --data_path /data2/user2/rl_tsf/All_data/processed_data/exchange_rate_336_192_train.csv \
    --epochs 100 --batch_size 16 --seq_len 336 --pred_len 192 --text_type original_text --data_name exchange_rate
python ./Models/TextFusionHTS/train_tfhts.py --data_path /data2/user2/rl_tsf/All_data/processed_data/exchange_rate_336_336_train.csv \
    --epochs 100 --batch_size 16 --seq_len 336 --pred_len 336 --text_type original_text --data_name exchange_rate