#!/bin/bash

python optimize.py \
    --study_name exchange_rate_96_336 \
    --root_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/others \
    --config exchange_rate_96_336.yaml \
    --data_name exchange_rate \
    --seq_len 96 \
    --pred_len 336 \
    --freq d