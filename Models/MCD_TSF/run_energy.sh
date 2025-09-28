export CUDA_VISIBLE_DEVICES=2

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/Time-MMD \
    --config energy_96_12.yaml \
    --data_name Energy \
    --seq_len 96 \
    --pred_len 12 \
    --text_len 36 \
    --freq w

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/Time-MMD \
    --config energy_96_24.yaml \
    --data_name Energy \
    --seq_len 96 \
    --pred_len 24 \
    --text_len 36 \
    --freq w

python -u Models/MCD_TSF/exe_forecasting.py\
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/Time-MMD \
    --config energy_96_48.yaml \
    --data_name Energy \
    --seq_len 96 \
    --pred_len 48 \
    --text_len 36 \
    --freq w
