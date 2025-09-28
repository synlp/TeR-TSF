#!/bin/bash

python optimize.py \
    --study_name weather_5_5 \
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/TTC \
    --config weather_5_5.yaml \
    --data_name weather \
    --seq_len 5 \
    --pred_len 5 \
    --freq d