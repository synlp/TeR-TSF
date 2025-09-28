export CUDA_VISIBLE_DEVICES=3

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/TTC \
    --config weather_5_5.yaml \
    --data_name weather \
    --seq_len 5 \
    --pred_len 5 \
    --text_len 36 \
    --freq d

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/TTC \
    --config weather_6_6.yaml \
    --data_name weather \
    --seq_len 6 \
    --pred_len 6 \
    --text_len 36 \
    --freq d

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/TTC \
    --config weather_7_7.yaml \
    --data_name weather \
    --seq_len 7 \
    --pred_len 7 \
    --text_len 36 \
    --freq d
