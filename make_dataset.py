import argparse
import pandas as pd
# from Modules.TEP import TextEhancementProvider
from Modules.BM import BaseMultimodalTSFModel
import ast
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import accelerate
import torch.distributed as dist
import torch
from utils.prepare4llm import get_desc
from utils.timefeatures import time_features
import re

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
    "exchange":{
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
        "mean": 0.0,
        "std": 1.0,
        "freq": 'h'
    }
}

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
    def __init__(self, df_all, size=None, data='Agriculture', text_type='reinforced_text'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        #
        self.df = df_all
        self.mean, self.std = stat_dict[data]["mean"], stat_dict[data]["std"]
        self.history_series = (np.array(self.df['history_series'].apply(ast.literal_eval).values.tolist()) - self.mean) / self.std
        self.horizon_series = (np.array(self.df['horizon_series'].apply(ast.literal_eval).values.tolist()) - self.mean) / self.std
        self.freq = stat_dict[data]["freq"]
        self.text_type = text_type
        self.desc = get_desc(data, self.seq_len, self.pred_len)
        self.tot_len = len(self.df)
    def extract_timestamps(self, data_str):

        start_marker = "Input data points:"
        end_marker = "Now generate the analysis result in the following format:"
        
        start_idx = data_str.find(start_marker)
        if start_idx == -1:
            return []
        
        start_idx += len(start_marker)
        
        end_idx = data_str.find(end_marker, start_idx)
        if end_idx == -1:
            return []
        
        data_block = data_str[start_idx:end_idx]
        
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
            'text_mark': txt_mark
        }

        return s, self.history_series[index], self.horizon_series[index], self.df.loc[index, self.text_type], self.df.loc[index, "prompt"]


class TFHTSDataset(Dataset):
    def __init__(self, df, seq_len, pred_len, text_type, tokenizer, text_model, data_name, device):
        self.df = df
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.device = device
        self.data_name = data_name
        self.text_type = text_type
        self.mean, self.std = stat_dict[self.data_name]["mean"], stat_dict[self.data_name]["std"]

        ts_data = [eval(ts) for ts in self.df['history_series']]
        pred_data = [eval(pred) for pred in self.df['horizon_series']]

        self.ts_data = (np.array(ts_data, dtype=np.float32) - self.mean) / self.std
        self.pred_data = (np.array(pred_data, dtype=np.float32) - self.mean) / self.std
        

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
            'ts': ts,
            'text_emb': text_emb,
            'pred': pred,
            'idx': idx,
            'prompt': self.df.loc[idx, "prompt"],
            'text': self.df.loc[idx, self.text_type]
        }