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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
    "climate": {
        "mean": 59.6538856,
        "std": 16.95276253
    },
    "Heart_Rate": {
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

class temp_dataset(Dataset):
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



def main():
    parser = argparse.ArgumentParser(description="LLM Prediction with Examples")
    parser.add_argument("--csv_file", type=str, help="Path to the CSV file", default="./reinforced_my_datasets/Time-MMD/Climate/Climate_96_12_train_0.csv")
    parser.add_argument("--hist_len", type=int, help="Historical time series length", default=96)
    parser.add_argument("--pred_len", type=int, help="Prediction length", default=12)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    parser.add_argument("--save_dir", type=str, help="dir to save dataset", default='./reward_my_datasets/Time-MMD/Climate')
    parser.add_argument("--text_type", type=str, help="", default='reinforced_text')
    args = parser.parse_args()

    df_all = pd.read_csv(args.csv_file)
    data = args.csv_file.split("/")[-1].split("_")[0]
    df_all = df_all.dropna(axis=0, how='any', subset=[args.text_type])
    data_set = temp_dataset(df_all, data, args.text_type)
    data_loader = DataLoader(data_set, args.batch_size, shuffle=False, num_workers=128, drop_last=False)

    tsf_model = BaseMultimodalTSFModel(args.hist_len, args.pred_len, bm_name="chattime")

    # accelerator = accelerate.Accelerator()
    # tsf_model, data_loader = accelerator.prepare(tsf_model, data_loader)

    all_data_with_reward = {
        'history_series': [],
        'horizon_series': [],
        'prompt': [],
        args.text_type: [],
        'reward1' : []
    }
    # for history_series, horizon_series, reinforced_texts in tqdm(data_loader, disable=not accelerator.is_local_main_process):
    for history_series, horizon_series, texts, prompts in tqdm(data_loader):
        pred_serie = tsf_model.predict(history_series, texts)

        # history_series = accelerator.gather_for_metrics(history_series)
        # horizon_series = accelerator.gather_for_metrics(horizon_series)
        # reinforced_texts = accelerator.gather_for_metrics(reinforced_texts)
        # pred_serie = accelerator.gather_for_metrics(pred_serie)
        
        all_data_with_reward['history_series'].extend(history_series.cpu().numpy().tolist())
        all_data_with_reward['horizon_series'].extend(horizon_series.cpu().numpy().tolist())
        all_data_with_reward['prompt'].extend(prompts)
        all_data_with_reward[args.text_type].extend(texts)

        pred_serie = np.concatenate(pred_serie).reshape(len(horizon_series), -1)
        reward = -np.mean((pred_serie - horizon_series.cpu().numpy()) ** 2, axis=1)
        all_data_with_reward['reward1'].extend(reward.tolist())

    # if accelerator.is_local_main_process:
    os.makedirs(args.save_dir, exist_ok=True)
    all_data_with_reward = pd.DataFrame(all_data_with_reward)
    all_data_with_reward.to_csv(os.path.join(args.save_dir, os.path.basename(args.csv_file)), index=False)

    # dist.barrier()  # 确保所有进程都停止于此
    # dist.destroy_process_group()

if __name__ == "__main__":
    main()