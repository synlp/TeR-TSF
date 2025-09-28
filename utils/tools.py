import torch
import gc
from torch.utils.data import Dataset
import ast
import numpy as np
import math
from utils.prepare4llm import get_desc
from utils.timefeatures import time_features
import re
import pandas as pd
from tqdm import tqdm
import os
import ahocorasick
import json
from collections import OrderedDict



stat_dict = {
    "Agriculture": {
        "mean": 144.59045661,
        "std": 23.02460005,
        "freq": 'm'
    },
    "Climate": {
        "mean": 56.13323596,
        "std": 10.09066259,
        "freq": 'w'
    },
    "Economy": {
        "mean": -32983.5974359,
        "std": 23032.45317885,
        "freq": 'm'
    },
    "Energy": {
        "mean": 2.12598414,
        "std": 0.97018236,
        "freq": 'w'
    },
    "Environment": {
        "mean": 87.56969155,
        "std": 42.80683806,
        "freq": 'd'
    },
    "Health_US": {
        "mean": 1.57049671,
        "std": 1.20796971,
        "freq": 'w'
    },
    "SocialGood": {
        "mean": 5.63603744,
        "std": 1.63104511,
        "freq": 'm'
    },
    "Traffic": {
        "mean": 171844.25274725,
        "std": 51870.70593244,
        "freq": 'm'
    },
    "ETTh1": {
        "mean": 16.29471487,
        "std": 8.34847203,
        "freq": 'h'
    },
    "exchange_rate":{
        "mean": 0.60482487,
        "std": 0.0952995,
        "freq": 'd'
    },
    "weather": {
        "mean": 59.6538856,
        "std": 16.95276253,
        "freq": 'd'
    },
    "Heart_Rate": {
        "mean": 160.59121974,
        "std": 9.57681761,
        "freq": 'd'
    },
    "MTBench_weather": {
        "mean": 15.826329,
        "std": 10.204743,
        "freq": 'h'
    },
    "MTBench_finance": {
        "mean": 0.0,  # 已标准化，均值接近0
        "std": 1.0,  # 已标准化，标准差接近1
        "freq": 'h'
    }
}

def clear_dataloder(dataloader):
    # 清理步骤 ----
    # 1. 终止工作进程 (关键！)
    # if hasattr(dataloader, '_workers'):  # PyTorch 1.7+ 的兼容写法
    #     for worker in dataloader._workers:
    #         if worker.is_alive():
    #             worker.terminate()  # 强制终止
    #     dataloader._workers = []
    if hasattr(dataloader, 'shutdown'):
        dataloader.shutdown()  # PyTorch 1.7+
    elif hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
        dataloader._iterator._shutdown_workers()  # PyTorch <1.7

    # 2. 删除 DataLoader 对象
    del dataloader

def downsample_dataset(df, ratio=None, desired_size=128):
    """
    通过调整滑窗步长缩减数据集大小
    
    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    ratio (float): 采样比例 (0 < ratio <= 1)
    desired_size (int): 期望的数据集大小
    """
    N = df.shape[0]  # N: 样本数量
    if N < desired_size:
        return df
    
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
    print(f"数据集样本数量从 {N} -> {len(sampled_df)}")
    return sampled_df


class gen_text_dataset(Dataset):
    def __init__(self, df_all, down_sample=False):
        super().__init__()
        if down_sample:
            df_all = downsample_dataset(df_all)
        self.prompts = df_all['prompt']
        self.history_series = df_all['history_series']
        self.horizon_series = df_all['horizon_series']

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx], self.history_series[idx], self.horizon_series[idx]

class pred_dataset(Dataset):
    def __init__(self, df_all, data, text_type='reinforced_text', down_sample=False):
        super().__init__()
        if down_sample:
            df_all = downsample_dataset(df_all)
        mean, std = stat_dict[data]["mean"], stat_dict[data]["std"]
        self.history_series = (np.array(df_all['history_series'].apply(ast.literal_eval).values.tolist()) - mean) / std
        self.horizon_series = (np.array(df_all['horizon_series'].apply(ast.literal_eval).values.tolist()) - mean) / std
        self.reinforced_text = np.array(df_all[text_type])
        self.prompt = np.array(df_all['prompt'])
    def __len__(self):
        return len(self.history_series)
    def __getitem__(self, idx):
        return self.history_series[idx], self.horizon_series[idx], self.reinforced_text[idx], self.prompt[idx]
    
class ChatTime_dataset(Dataset):
    def __init__(self, df_all, data, text_type='reinforced_text'):
        super().__init__()
        mean, std = stat_dict[data]["mean"], stat_dict[data]["std"]
        self.history_series = (np.array(df_all['history_series'].apply(ast.literal_eval).values.tolist()) - mean) / std
        self.horizon_series = (np.array(df_all['horizon_series'].apply(ast.literal_eval).values.tolist()) - mean) / std
        self.reinforced_text = np.array(df_all[text_type])
        self.prompt = np.array(df_all['prompt'])
    def __len__(self):
        return len(self.history_series)
    def __getitem__(self, idx):
        return self.history_series[idx], self.horizon_series[idx], self.reinforced_text[idx], self.prompt[idx]

class MCD_dataset(Dataset):
    def __init__(self, df_all, size=None, data='Agriculture', text_type='reinforced_text', down_sample=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        #
        if down_sample:
            df_all = downsample_dataset(df_all)
        self.df = df_all
        self.mean, self.std = stat_dict[data]["mean"], stat_dict[data]["std"]
        self.history_series = (np.array(self.df['history_series'].apply(ast.literal_eval).values.tolist()) - self.mean) / self.std
        self.horizon_series = (np.array(self.df['horizon_series'].apply(ast.literal_eval).values.tolist()) - self.mean) / self.std
        self.freq = stat_dict[data]["freq"]
        self.text_type = text_type
        self.desc = get_desc(data, self.seq_len, self.pred_len)
        self.tot_len = len(self.df)
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
        
        # 使用正则表达式匹配所有时间戳（格式：YYYY-MM-DD）
        timestamp_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
        timestamps = re.findall(timestamp_pattern, data_block)
    
        return timestamps
    def __len__(self):
        return len(self.history_series)
    def __getitem__(self, index):
        seq_x = self.history_series[index].reshape(-1, 1)
        seq_y = self.horizon_series[index].reshape(-1, 1)
        seq_x_txt = self.df.loc[index, self.text_type]
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
            'text_mark': txt_mark,
            'history_series': self.history_series[index],
            'horizon_series': self.horizon_series[index],
            'prompt': self.df.loc[index, "prompt"],
            self.text_type: self.df.loc[index, self.text_type]
        }

        return s


class TFHTSDataset(Dataset):
    def __init__(self, df, seq_len, pred_len, text_type, tokenizer, text_model, data_name, device, down_sample=False):
        self.df = df
        if down_sample:
            df = downsample_dataset(df)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.device = device
        self.data_name = data_name
        self.text_type = text_type
        self.mean, self.std = stat_dict[self.data_name]["mean"], stat_dict[self.data_name]["std"]
        # 预处理时间序列数据
        ts_data = [eval(ts) for ts in self.df['history_series']]
        pred_data = [eval(pred) for pred in self.df['horizon_series']]
        # import pdb; pdb.set_trace()
        self.ts_data = (np.array(ts_data, dtype=np.float32) - self.mean) / self.std
        self.pred_data = (np.array(pred_data, dtype=np.float32) - self.mean) / self.std
        
        # 预处理文本数据并提取嵌入
        print("Extracting text embeddings...")
        self.text_embeddings = self._extract_text_embeddings()
    
    def _extract_text_embeddings(self):
        embeddings = []
        texts = self.df[self.text_type].astype(str).tolist()
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Processing texts"):
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                      truncation=True, max_length=512).to(self.device)
                outputs = self.text_model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                embedding = hidden.mean(dim=1).cpu().numpy()[0]
                embeddings.append(embedding)
                
        return np.array(embeddings)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ts = self.ts_data[idx]
        pred = self.pred_data[idx]
        text_emb = self.text_embeddings[idx].astype(np.float32)
        
        return {
            'history_series': ts,
            'text_emb': text_emb,
            'horizon_series': pred,
            'idx': idx,
            'prompt': self.df.loc[idx, "prompt"],
            self.text_type: self.df.loc[idx, self.text_type],
        }
    
class Config:
    def __init__(self, **kwargs):
        # 设置默认参数
        self.task_name = 'long_term_forecast'
        self.model_id = 'custom_model'
        self.model_comment = 'train'
        self.seed = 2021
        
        self.data_path = None
        self.features = 'S'
        self.checkpoints = './Models/Time_LLM/checkpoints/'
        
        self.seq_len = 36
        self.label_len = 0
        self.pred_len = 6
        
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        self.d_model = 16
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 32
        self.dropout = 0.1
        self.patch_len = 16
        self.stride = 8
        self.llm_model = 'LLAMA'
        self.llm_dim = 4096
        self.llm_layers = 6
        
        self.num_workers = 2
        self.train_epochs = 10
        self.batch_size = 8
        self.patience = 5
        self.learning_rate = 0.0001
        self.percent = 100
        self.device = 'auto'
        # 覆盖用户提供的参数值
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")
    def __repr__(self):
        # 打印所有配置值（可选）
        params = [f"{k}={v}" for k, v in vars(self).items()]
        return f"Config({', '.join(params)})"


class Time_LLM_Dataset(Dataset):
    def __init__(self, df_raw, seq_len, pred_len, data_name, text_type="reinforced_text"):
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
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.text_type = text_type
        self.data_name = data_name

        self.df = df_raw
        self.__read_data__()
        
        self.enc_in = 1  # 单变量时间序列
        self.c_out = 1   # 输出维度
        
    def __read_data__(self):
        
        # 解析字符串格式的列表
        def parse_series(series_str):
            try:
                return ast.literal_eval(series_str)
            except:
                # 如果解析失败，尝试其他方法
                series_str = series_str.strip('[]')
                return [float(x.strip()) for x in series_str.split(',')]
            
        self.mean, self.std = stat_dict[self.data_name]["mean"], stat_dict[self.data_name]["std"]
            
        # 预处理时间序列数据
        history_series = [eval(ts) for ts in self.df['history_series']]
        horizon_series = [eval(pred) for pred in self.df['horizon_series']]
        # import pdb; pdb.set_trace()
        self.history_series = (np.array(history_series, dtype=np.float32) - self.mean) / self.std
        self.horizon_series = (np.array(horizon_series, dtype=np.float32) - self.mean) / self.std
        
            
    def __getitem__(self, index):
        """获取单个样本"""
        # 历史序列 [seq_len] -> [seq_len, 1]
        seq_x = self.history_series[index].reshape(-1, 1)
        
        # 目标序列，包含label_len部分 [label_len + pred_len, 1]

        seq_y = self.horizon_series[index].reshape(-1, 1)
        
        # 时间特征标记 (简化为全零)
        seq_x_mark = np.zeros((len(seq_x), 1))
        seq_y_mark = np.zeros((len(seq_y), 1))
        
        # 获取prompt
        prompt = self.df.loc[index, "prompt"]

        s = {
            "history_series": seq_x,
            "horizon_series": seq_y,
            "seq_x_mark": seq_x_mark,
            "seq_y_mark": seq_y_mark,
            "prompt": prompt,
            self.text_type: self.df.loc[index, self.text_type]

        }
        
        return s
    
    def __len__(self):
        return len(self.history_series)



def calculate_normalized_scores(text_input):
    """
    计算文本列表中每个文本的归一化分数
    :param vocab_list: 词汇列表，包含需要检测的单词
    :param text_list: 文本列表，包含待检测的字符串
    :return: 归一化分数列表，每个元素对应text_list中相应文本的分数
    """
    # 处理空词汇表的边界情况
    # if not vocab_list:
    #     return [0.0] * len(text_list)
    vocab_list = [
        "trend", "linear", "non-linear", "seasonality", "peak", "turning point",
        "stable", "disaster", "holiday", "impact", "long-term", "short-term",
        "business", "anomalies", "event", "periodicity", "affect", "up",
        "down", "drift", "steady", "fluctuation", "reason", "because", "phase", "outlier",
        "decrease", "increase", "factor", "pattern", "seasonal"
    ]
    
    # 预处理：转换为小写并过滤空字符串
    vocab_list = [word.strip().lower() for word in vocab_list]
    vocab_list = [word for word in vocab_list if word]
    
    # 构建Aho-Corasick自动机
    automaton = ahocorasick.Automaton()
    unique_words = set()
    for word in vocab_list:
        # 只添加唯一的单词到自动机
        if word not in unique_words:
            automaton.add_word(word, word)
            unique_words.add(word)
    
    automaton.make_automaton()
    total_vocab_count = len(vocab_list)  # 原始词汇表长度（含重复）
    
    if isinstance(text_input, list):
        # 计算每个文本的分数
        scores = []
        for text in text_input:
            text = text.lower()
            matched_words = set()
            
            # 在文本中搜索所有匹配的词汇
            for _, word in automaton.iter(text):
                matched_words.add(word)
            
            # 计算归一化分数
            score = len(matched_words) / total_vocab_count
            scores.append(score)
        return scores
    elif isinstance(text_input, str):
        text = text_input.lower()
        matched_words = set()
        for _, word in automaton.iter(text):
            matched_words.add(word)
        score = len(matched_words) / total_vocab_count
        return score
    
class RecordExpMetrics:
    def __init__(self, file_path):
        self.file_path = file_path

    def is_exist(self, fixed_args, varying_args):
        # 将固定参数和变动参数转换为可哈希的元组键
        fixed_key = tuple(sorted(fixed_args.items()))
        varying_key = tuple(sorted(varying_args.items()))
        # 尝试读取现有数据
        existing_data = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                # 将字符串键转换回元组
                raw_data = json.load(f)
                existing_data = {
                    ast.literal_eval(k): {
                        ast.literal_eval(inner_k): inner_v 
                        for inner_k, inner_v in v.items()
                    }
                    for k, v in raw_data.items()
                }
        if fixed_key in existing_data:
            if varying_key in existing_data[fixed_key]:
                return True  # 记录已存在
        return False
    
    def add_result(self, fixed_args, varying_args, metrics, overwrite=False):
        # 将固定参数和变动参数转换为可哈希的元组键
        fixed_key = tuple(sorted(fixed_args.items()))
        varying_key = tuple(sorted(varying_args.items()))
        # 尝试读取现有数据
        existing_data = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                # 将字符串键转换回元组
                raw_data = json.load(f)
                existing_data = {
                    ast.literal_eval(k): {
                        ast.literal_eval(inner_k): inner_v 
                        for inner_k, inner_v in v.items()
                    }
                    for k, v in raw_data.items()
                }

        if fixed_key in existing_data:
            if varying_key in existing_data[fixed_key]:
                print("记录已经存在，跳过...")
                return
        else:
            existing_data[fixed_key] = {}

        existing_data[fixed_key][varying_key] = metrics
        
        # 将元组键转换为字符串以便JSON序列化
        serializable_data = {
            str(fixed_key): {
                str(inner_key): inner_value 
                for inner_key, inner_value in inner_dict.items()
            }
            for fixed_key, inner_dict in existing_data.items()
        }
        
        # 写入文件
        # import pdb; pdb.set_trace()
        with open(self.file_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)

        print("评测结果写入成功！")