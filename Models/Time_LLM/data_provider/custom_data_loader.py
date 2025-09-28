import os
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom_CSV(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='custom_data.csv', 
                 scale=True, percent=100):
        """
        自定义CSV数据集加载器
        Args:
            root_path: 数据根目录
            flag: 'train', 'val', 'test'
            size: [seq_len, label_len, pred_len]
            data_path: CSV文件路径
            scale: 是否标准化
            percent: 使用数据的百分比
        """
        if size is None:
            self.seq_len = 36
            self.label_len = 0  # 数据已明确分离输入输出，无需label_len
            self.pred_len = 6
        else:
            self.seq_len = size[0]
            self.label_len = size[1] 
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val', 'all']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        self.enc_in = 1  # 单变量时间序列
        self.c_out = 1   # 输出维度
        
    def __read_data__(self):
        """读取CSV数据"""
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 解析字符串格式的列表
        def parse_series(series_str):
            try:
                return ast.literal_eval(series_str)
            except:
                # 如果解析失败，尝试其他方法
                series_str = series_str.strip('[]')
                return [float(x.strip()) for x in series_str.split(',')]
        
        # 解析历史序列和目标序列
        history_series = df_raw['history_series'].apply(parse_series).tolist()
        horizon_series = df_raw['horizon_series'].apply(parse_series).tolist()
        prompts = df_raw['prompt'].tolist()
        
        # 转换为numpy数组
        history_data = np.array(history_series)  # [num_samples, seq_len]
        horizon_data = np.array(horizon_series)  # [num_samples, pred_len]
        
        # 数据分割
        num_train = int(len(history_data) * 0.7)
        num_test = int(len(history_data) * 0.2) 
        num_vali = len(history_data) - num_train - num_test
        
        border1s = [0, num_train, num_train + num_vali, 0]  # 'all'使用所有数据
        border2s = [num_train, num_train + num_vali, len(history_data), len(history_data)]  # 'all'使用所有数据
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 应用百分比
        if self.set_type == 0:  # train
            border2 = border1 + int((border2 - border1) * self.percent // 100)
        
        # 提取当前split的数据
        self.history_data = history_data[border1:border2]
        self.horizon_data = horizon_data[border1:border2]
        self.prompts = prompts[border1:border2]
        
        # 标准化
        if self.scale:
            # 使用训练集计算scaler
            train_history = history_data[border1s[0]:border2s[0]]
            train_horizon = horizon_data[border1s[0]:border2s[0]]
            
            # 合并历史和未来数据来计算统计量
            all_train_data = np.concatenate([train_history, train_horizon], axis=1)
            all_train_data_flat = all_train_data.flatten().reshape(-1, 1)
            
            self.scaler = StandardScaler()
            self.scaler.fit(all_train_data_flat)
            
            # 标准化当前数据
            history_flat = self.history_data.flatten().reshape(-1, 1)
            horizon_flat = self.horizon_data.flatten().reshape(-1, 1)
            
            history_scaled = self.scaler.transform(history_flat)
            horizon_scaled = self.scaler.transform(horizon_flat)
            
            self.history_data = history_scaled.reshape(self.history_data.shape)
            self.horizon_data = horizon_scaled.reshape(self.horizon_data.shape)
        else:
            self.scaler = None
            
    def __getitem__(self, index):
        """获取单个样本"""
        # 历史序列 [seq_len] -> [seq_len, 1]
        seq_x = self.history_data[index].reshape(-1, 1)
        
        # 目标序列，包含label_len部分 [label_len + pred_len, 1]
        if self.label_len > 0:
            # 如果需要label，从历史序列末尾取
            label_part = seq_x[-self.label_len:] 
            horizon_part = self.horizon_data[index].reshape(-1, 1)
            seq_y = np.concatenate([label_part, horizon_part], axis=0)
        else:
            seq_y = self.horizon_data[index].reshape(-1, 1)
        
        # 时间特征标记 (简化为全零)
        seq_x_mark = np.zeros((len(seq_x), 1))
        seq_y_mark = np.zeros((len(seq_y), 1))
        
        # 获取prompt
        prompt = self.prompts[index]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt
    
    def __len__(self):
        return len(self.history_data)
    
    def inverse_transform(self, data):
        """反标准化"""
        if self.scaler is None:
            return data
        
        # data shape: [batch_size, seq_len, features]
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        data_inv = self.scaler.inverse_transform(data_flat)
        return data_inv.reshape(original_shape) 