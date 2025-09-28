from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import os
from glob import glob
"""
Time-MMD
"""
# root_path = "/home/suchen/Time-MMD-main/"
# data_path = "Traffic/Traffic.csv"
# seq_len = 36

# scaler = StandardScaler()
# df_num = pd.read_csv(os.path.join(root_path, 'numerical', data_path))
# df_num = df_num.dropna(axis='index', how='any', subset=['OT'])
# df_num['date'] = pd.to_datetime(df_num['date'])
# df_num = df_num.sort_values('date', ascending=True).reset_index(drop=True)
# num_train = int(len(df_num) * 0.7)
# num_test = int(len(df_num) * 0.2)
# num_vali = len(df_num) - num_train - num_test
# border1s = [0, num_train - seq_len, len(df_num) - num_test - seq_len]
# border2s = [num_train, num_train + num_vali, len(df_num)]
# df_data = df_num[['OT']]
# train_data = df_data[border1s[0]:border2s[0]]
# scaler.fit(train_data.values)
# data1 = scaler.transform(df_data.values).astype(np.float32)
# mean_data = scaler.mean_
# std_data = scaler.scale_
# data2 = (df_data.values - mean_data)/std_data

# print(f"mean: {mean_data}, std: {std_data}")
# print(np.sum((data1-data2)**2))

"""
ETTh1, exchange
"""
# root_path = "/home/suchen/all_six_datasets/"
# data_path = "exchange_rate/exchange_rate.csv"
# seq_len = 96
# df_raw = pd.read_csv(os.path.join(root_path, data_path))

# # border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
# # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

# num_train = int(len(df_raw) * 0.7)
# num_test = int(len(df_raw) * 0.2)
# num_vali = len(df_raw) - num_train - num_test
# border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
# border2s = [num_train, num_train + num_vali, len(df_raw)]

# df_data = df_raw[['OT']]
# scaler = StandardScaler()
# train_data = df_data[border1s[0]:border2s[0]]
# scaler.fit(train_data.values)
# data = scaler.transform(df_data.values)
# mean_data = scaler.mean_
# std_data = scaler.scale_
# print(f"mean: {mean_data}, std: {std_data}")

"""
TTC
"""
# weather
# data_path = "/home/suchen/Multimodal_Forecasting-main/data/climate_2014_2023_final.csv"
# hist_len = 7
# df_all = pd.read_csv(data_path)
# df_all = df_all.dropna(subset=["date", "temp"]).reset_index(drop=True)

# num_train = int(len(df_all) * 0.7)
# num_test = int(len(df_all) * 0.2)
# num_vali = len(df_all) - num_train - num_test
# border1s = [0, num_train - hist_len, len(df_all) - num_test - hist_len]
# border2s = [num_train, num_train + num_vali, len(df_all)]
# df_data = df_all[['temp']]
# scaler = StandardScaler()
# train_data = df_data[border1s[0]:border2s[0]]
# scaler.fit(train_data.values)
# data = scaler.transform(df_data.values)
# mean_data = scaler.mean_
# std_data = scaler.scale_
# print(f"mean: {mean_data}, std: {std_data}")

# medical
csv_file_dir = "/home/suchen/Multimodal_Forecasting-main/data"
var_name = "Heart_Rate"
hist_len = 6
data_dir = os.path.join(csv_file_dir, 'medical')
csv_list = glob(data_dir+'/*.csv')
num_train = int(len(csv_list) * 0.7)
num_test = int(len(csv_list) * 0.2)
train_csv_path_list = csv_list[:num_train]

new_data = []
for data_path in train_csv_path_list:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["date", var_name]).reset_index(drop=True)
    new_data.append(df[[var_name]].values)
new_data = np.concatenate(new_data, axis=0)
scaler = StandardScaler()
scaler.fit(new_data)
data = scaler.transform(new_data)
mean_data = scaler.mean_
std_data = scaler.scale_
print(f"mean: {mean_data}, std: {std_data}")