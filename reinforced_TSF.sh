python reinforced_prediction.py \
    --csv_file ./reinforced_my_datasets/Time-MMD/Climate/Climate_96_12_train_0.csv \
    --hist_len 96 \
    --pred_len 12 \
    --batch_size 8 \
    --save_dir ./reward_my_datasets/Time-MMD/Climate

python reinforced_prediction.py \
    --csv_file ./reinforced_my_datasets/Time-MMD/Climate/Climate_96_12_train_1.csv \
    --hist_len 96 \
    --pred_len 12 \
    --batch_size 8 \
    --save_dir ./reward_my_datasets/Time-MMD/Climate
