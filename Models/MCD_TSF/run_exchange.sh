export CUDA_VISIBLE_DEVICES=4

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/others \
    --config exchange_rate_96_96.yaml \
    --data_name exchange_rate \
    --seq_len 96 \
    --pred_len 96 \
    --text_len 36 \
    --freq d

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/others \
    --config exchange_rate_96_192.yaml \
    --data_name exchange_rate \
    --seq_len 96 \
    --pred_len 192 \
    --text_len 36 \
    --freq d

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/others \
    --config exchange_rate_96_336.yaml \
    --data_name exchange_rate \
    --seq_len 96 \
    --pred_len 336 \
    --text_len 36 \
    --freq d
