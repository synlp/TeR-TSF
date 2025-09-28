#!/bin/bash

python optimize.py \
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/Time-MMD \
    --study_name Energy_96_24 \
    --config energy_96_24.yaml \
    --data_name Energy \
    --seq_len 96 \
    --pred_len 24 \
    --freq w