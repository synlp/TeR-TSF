#!/bin/bash

python optimize.py \
    --study_name weather_6_6 \
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/TTC \
    --config weather_6_6.yaml \
    --data_name weather \
    --seq_len 6 \
    --pred_len 6 \
    --freq d