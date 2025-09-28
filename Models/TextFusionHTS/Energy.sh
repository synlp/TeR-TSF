export CUDA_VISIBLE_DEVICES=1

python train_tfhts.py --data_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/Time-MMD/Energy/Energy_96_12_train.csv \
    --epochs 100 --batch_size 16 --seq_len 96 --pred_len 12 --text_type original_text --data_name Energy
python train_tfhts.py --data_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/Time-MMD/Energy/Energy_96_24_train.csv \
    --epochs 100 --batch_size 16 --seq_len 96 --pred_len 24 --text_type original_text --data_name Energy
python train_tfhts.py --data_path /media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/Time-MMD/Energy/Energy_96_48_train.csv \
    --epochs 100 --batch_size 16 --seq_len 96 --pred_len 48 --text_type original_text --data_name Energy