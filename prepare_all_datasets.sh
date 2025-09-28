### 原始数据集位置设置
time_mmd_dir=/data2/user2/Time-MMD-main
ttc_dir=/data2/user2/TTC
etth1=/data2/user2/all_six_datasets/ETT-small/ETTh1.csv
exchange=/data2/user2/all_six_datasets/exchange_rate/exchange_rate.csv
### 处理后的保存位置
processed_timeMMD_dir=/data2/user2/ter_tsf/processed_timeMMD
postprocessed_datasets_save_dir=/data2/user2/ter_tsf/processed_data

# python get_MTBench_weather_with_prompt.py \
#     --hist_len 96 \
#     --pred_len 12 

# python get_MTBench_weather_with_prompt.py \
#     --hist_len 96 \
#     --pred_len 24 

# python get_MTBench_weather_with_prompt.py \
#     --hist_len 96 \
#     --pred_len 48 

python get_MTBench_finance_with_prompt.py \
    --hist_len 96 \
    --pred_len 12 

python get_MTBench_finance_with_prompt.py \
    --hist_len 96 \
    --pred_len 24 

python get_MTBench_finance_with_prompt.py \
    --hist_len 96 \
    --pred_len 48 


## 预处理Time-MMD数据集
# python preprocess_time_mmd.py \
#     --dataset_dir $time_mmd_dir \
#     --save_dir $processed_timeMMD_dir


# ## 将预处理的Time-MMD数据集再构建构成prompt，并划分成训练集、验证集和测试集
# for pred_len in 6 12 18
# do
#     python get_timeMMD_with_prompt.py \
#         --csv_file_dir $processed_timeMMD_dir \
#         --domain Agriculture \
#         --hist_len 36 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

# for pred_len in 12 24 48
# do
#     python get_timeMMD_with_prompt.py \
#         --csv_file_dir $processed_timeMMD_dir \
#         --domain Climate \
#         --hist_len 96 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

# for pred_len in 6 12 18
# do
#     python get_timeMMD_with_prompt.py \
#         --csv_file_dir $processed_timeMMD_dir \
#         --domain Economy \
#         --hist_len 36 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

# for pred_len in 12 24 48
# do
#     python get_timeMMD_with_prompt.py \
#         --csv_file_dir $processed_timeMMD_dir \
#         --domain Energy \
#         --hist_len 96 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

# for pred_len in 48 96 192
# do
#     python get_timeMMD_with_prompt.py \
#         --csv_file_dir $processed_timeMMD_dir \
#         --domain Environment \
#         --hist_len 336 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

# for pred_len in 12 24 48
# do
#     python get_timeMMD_with_prompt.py \
#         --csv_file_dir $processed_timeMMD_dir \
#         --domain Health_US \
#         --hist_len 96 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

# for pred_len in 6 12 18
# do
#     python get_timeMMD_with_prompt.py \
#         --csv_file_dir $processed_timeMMD_dir \
#         --domain SocialGood \
#         --hist_len 36 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

# for pred_len in 6 12 18
# do
#     python get_timeMMD_with_prompt.py \
#         --csv_file_dir $processed_timeMMD_dir \
#         --domain Traffic \
#         --hist_len 36 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

## 将TTC的Weather划分成训练集、验证集和测试集
# for pred_len in 3 7 14
# do
#     python get_TTC_weather_with_prompt.py \
#         --csv_file_dir $ttc_dir \
#         --data climate_2014_2023_final \
#         --hist_len 14 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done

# # 将TTC的Medical划分成训练集、验证集和测试集
# for var_name in Respiratory_Rate Heart_Rate #SaO2 FiO2
# do
# for pred_len in 3 7 14
# do
#     python get_TTC_medical_with_prompt.py \
#         --csv_file_dir $ttc_dir \
#         --var_name $var_name \
#         --hist_len 14 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done
# done

# 将ETTh1和Exchange划分成训练集、验证集和测试集
# for pred_len in 96 192 336
# do
#     python get_other_dataset_prompt.py \
#         --csv_file_path $etth1 \
#         --data ETTh1 \
#         --hist_len 336 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir

#     python get_other_dataset_prompt.py \
#         --csv_file_path $exchange \
#         --data exchange_rate \
#         --hist_len 336 \
#         --pred_len $pred_len \
#         --save_dir $postprocessed_datasets_save_dir
# done