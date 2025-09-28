#!/bin/bash

python optimize.py \
    --study_name weather_7_7 \
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/TTC \
    --config weather_7_7.yaml \
    --data_name weather \
    --seq_len 7 \
    --pred_len 7 \
    --freq d