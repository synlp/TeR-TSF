import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings
from utils.prepare4llm import get_desc
import re
import ast
import math

warnings.filterwarnings('ignore')


stat_dict = {
    "Agriculture": {
        "mean": 144.59045661,
        "std": 23.02460005,
    },
    "Climate": {
        "mean": 56.13323596,
        "std": 10.09066259,
    },
    "Economy": {
        "mean": -32983.5974359,
        "std": 23032.45317885,
    },
    "Energy": {
        "mean": 2.12598414,
        "std": 0.97018236,
    },
    "Environment": {
        "mean": 87.56969155,
        "std": 42.80683806,
    },
    "Health": {
        "mean": 1.57049671,
        "std": 1.20796971
    },
    "SocialGood": {
        "mean": 5.63603744,
        "std": 1.63104511
    },
    "Traffic": {
        "mean": 171844.25274725,
        "std": 51870.70593244
    },
    "ETTh1": {
        "mean": 16.29471487,
        "std": 8.34847203
    },
    "exchange_rate":{
        "mean": 0.60482487,
        "std": 0.0952995
    },
    "weather": {
        "mean": 59.6538856,
        "std": 16.95276253
    },
    "medical": {
        "mean": 160.59121974,
        "std": 9.57681761
    },
    "MTBench_weather": {
        "mean": 15.826329,
        "std": 10.204743
    },
    "MTBench_finance": {
        "mean": 0.0,  # 已标准化，均值接近0
        "std": 1.0   # 已标准化，标准差接近1
    }
}

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 text_len=1, scaler_type='standard'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.text_len = text_len

        self.root_path = root_path
        self.data_path = data_path
        self.data_prefix = data_path.split('.')[0]
        
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type
        self.__read_data__()
        self.domain = data_path.split('/')[0]
        self.desc = get_desc(self.domain, self.seq_len, self.pred_len)
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        

    def __read_data__(self):
        df_num = pd.read_csv(os.path.join(self.root_path, 'numerical', self.data_path))
        df_report = pd.read_csv(os.path.join(self.root_path, 'textual', self.data_prefix + '_report.csv'))
        df_search = pd.read_csv(os.path.join(self.root_path, 'textual', self.data_prefix + '_search.csv'))

        df_num = df_num.dropna(axis='index', how='any', subset=['OT'])
        df_report = df_report.dropna(axis='index', how='any', subset=['fact'])
        df_search = df_search.dropna(axis='index', how='any', subset=['fact'])

        df_num['date'], df_num['start_date'], df_num['end_date'] = pd.to_datetime(df_num['date']), pd.to_datetime(df_num['start_date']), pd.to_datetime(df_num['end_date'])
        df_report['start_date'], df_report['end_date'] = pd.to_datetime(df_report['start_date']), pd.to_datetime(df_report['end_date'])

        df_num = df_num.sort_values('date', ascending=True).reset_index(drop=True)
        df_report = df_report.sort_values('start_date', ascending=True).reset_index(drop=True)
        num_train = int(len(df_num) * 0.7)
        num_test = int(len(df_num) * 0.2)
        num_vali = len(df_num) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_num) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_num)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        
        first_start_date = df_num.start_date[border1]
        final_end_date = df_num.end_date[border2-1]

        df_data = df_num[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
            self.mean_data = self.scaler.mean_
            self.std_data = self.scaler.scale_
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_num[['date']][border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


        self.data_stamp = data_stamp
        self.num_dates = df_num[['start_date', 'end_date']][border1:border2].reset_index(drop=True)
        self.txt_report = df_report[['start_date', 'end_date', 'fact']].loc[(df_report.end_date >= first_start_date) & (df_report.end_date <= final_end_date)]

    def collect_text(self, start_date, end_date):
        report = self.txt_report.loc[(self.txt_report.end_date >= start_date) & (self.txt_report.end_date <= end_date)]
        def add_datemark(row):
            return row['start_date'].strftime("%Y-%m-%d") + " to " + row['end_date'].strftime("%Y-%m-%d") + ": " + row['fact']
        if not report.empty:
            report = report.apply(add_datemark, axis=1).to_list()
            report.insert(0, self.desc)
            text_mark = 1
        else:
            report = ['NA']
            text_mark = 0
        all_txt = ' '.join(report)
        return all_txt, text_mark
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
    
        seq_x_stamp = self.data_stamp[s_begin:s_end]
        seq_y_stamp = self.data_stamp[r_begin:r_end]

        text_begin = s_end - self.text_len
        text_end = s_end

        seq_x_txt, txt_mark = self.collect_text(self.num_dates.start_date[text_begin], self.num_dates.end_date[text_end])

        observed_data = np.concatenate([seq_x, seq_y], axis=0)
        timesteps = np.concatenate([seq_x_stamp, seq_y_stamp], axis=0)
        observed_mask = np.ones_like(observed_data)
        gt_mask = np.concatenate([np.ones_like(seq_x), np.zeros_like(seq_y)], axis=0)

        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len).astype(np.float32), 
            'feature_id': np.arange(seq_x.shape[1]).astype(np.float32),
            'timesteps': timesteps,
            'texts': seq_x_txt,
            'text_mark': txt_mark
        }

        return s

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_Pretrain(Dataset):
    def __init__(self, root_path, size=None, data='Agriculture', freq=None, flag='train'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        #
        self.data = data
        data_ = data

        if data == 'exchange_rate' or data == 'ETTh1':
            df = pd.read_csv(f"{root_path}/{data_}_{self.seq_len}_{self.pred_len}_{flag}.csv")
        else:
            df = pd.read_csv(f"{root_path}/{data}/{data_}_{self.seq_len}_{self.pred_len}_{flag}.csv")
        # 小数据集调参
        # if flag == 'train':
        #     self.df = self.downsample_dataset(df, desired_size=1000)
        # elif flag == 'test':
        #     self.df = self.downsample_dataset(df, desired_size=500)
        # else:
        #     self.df = df
        self.df = df
        self.mean, self.std = stat_dict[data]["mean"], stat_dict[data]["std"]
        self.history_series = (np.array(self.df['history_series'].apply(ast.literal_eval).values.tolist()) - self.mean) / self.std
        self.horizon_series = (np.array(self.df['horizon_series'].apply(ast.literal_eval).values.tolist()) - self.mean) / self.std
        if freq == None:
            self.freq = stat_dict[data]["freq"]
        else:
            self.freq = freq
        
        self.desc = get_desc(data, self.seq_len, self.pred_len)
        self.tot_len = len(self.df)

    def downsample_dataset(self, df, ratio=None, desired_size=1000):
        """
        通过调整滑窗步长缩减数据集大小
        
        参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
        ratio (float): 采样比例 (0 < ratio <= 1)
        desired_size (int): 期望的数据集大小
        """
        # 读取原始数据
        # df = pd.read_csv(input_file, header=None)
        N, L = df.shape  # N: 样本数量, L: 时间序列长度
        
        # 验证输入参数
        if ratio is None and desired_size is None:
            raise ValueError("必须提供ratio或desired_size参数")
        
        if ratio is not None:
            if not (0 < ratio <= 1):
                raise ValueError("ratio必须在(0,1]范围内")
            desired_size = max(1, int(ratio * N))  # 确保至少有一个样本
        
        desired_size = min(desired_size, N)  # 不能超过原始大小
        
        # 计算需要的步长
        stride = max(1, math.ceil(N / desired_size))
        
        # 生成采样索引
        indices = list(range(0, N, stride))
        sampled_df = df.iloc[indices]
        sampled_df.reset_index(drop=True, inplace=True)
        return sampled_df
        
        # # 保存结果
        # sampled_df.to_csv(output_file, index=False, header=False)
        
        # # 打印采样信息
        # print(f"原始数据集大小: {N}条")
        # print(f"采样后数据集大小: {len(sampled_df)}条")
        # print(f"使用步长: {stride}")
        # print(f"采样比例: {len(sampled_df)/N:.4f}")

    
    def extract_timestamps(self, data_str):
        # 定义起始和结束标记
        start_marker = "Input data points:"
        end_marker = "Now generate the analysis result in the following format:"
        
        # 查找起始标记位置
        start_idx = data_str.find(start_marker)
        if start_idx == -1:
            return []
        
        # 计算数据块起始位置（跳过标记本身）
        start_idx += len(start_marker)
        
        # 查找结束标记位置
        end_idx = data_str.find(end_marker, start_idx)
        if end_idx == -1:
            return []
        
        # 提取两个标记之间的文本块
        data_block = data_str[start_idx:end_idx]
        
        # 使用正则表达式匹配所有时间戳（格式：YYYY-MM-DD HH:MM:SS）
        timestamp_pattern = r'\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b'
        timestamps = re.findall(timestamp_pattern, data_block)
    
        return timestamps
    
    def __getitem__(self, index):
        seq_x = self.history_series[index].reshape(-1, 1)
        seq_y = self.horizon_series[index].reshape(-1, 1)
        seq_x_txt = self.df.loc[index, "original_text"]
        if seq_x_txt == "No text.":
            seq_x_txt = "NA"
            txt_mark = 0
        else:
            seq_x_txt = self.desc + " " + seq_x_txt
            txt_mark = 1
        timesteps = self.extract_timestamps(self.df.loc[index, "prompt"])
        timesteps = time_features(pd.to_datetime(timesteps), freq=self.freq).transpose(1, 0)

        observed_data = np.concatenate([seq_x, seq_y], axis=0)
        observed_mask = np.ones_like(observed_data)
        gt_mask = np.concatenate([np.ones_like(seq_x), np.zeros_like(seq_y)], axis=0)

        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len).astype(np.float32), 
            'feature_id': np.arange(seq_x.shape[1]).astype(np.float32),
            'timesteps': timesteps,
            'texts': seq_x_txt,
            'text_mark': txt_mark
        }

        return s

    def __len__(self):
        return self.tot_len


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', target='OT',
                 scale=True, timeenc=0, freq='h', text_len=None,
                 scaler_type='minmax'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv', target='OT',
                 scale=True, timeenc=0, freq='t', text_len=None,
                 scaler_type='minmax'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        observed_data = np.concatenate(seq_x, seq_y, axis=0)
        timesteps = np.concatenate(seq_x_mark, seq_y_mark, axis=0)
        observed_mask = np.ones_like(observed_data)
        gt_mask = np.concatenate(np.ones_like(seq_x), np.zeros_like(seq_y), axis=0)

        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len) * 1.0, 
            'feature_id': np.arange(self.seq_x.shape[1]) * 1.0,
            'timesteps': timesteps

        }

        return s

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)