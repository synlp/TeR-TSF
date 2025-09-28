import argparse
import pandas as pd
from Modules.TeR import TextEhancementProvider
import ast
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import accelerate
import torch.distributed as dist

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class temp_dataset(Dataset):
    def __init__(self, df_all):
        super().__init__()
        self.prompts = df_all['prompt']
        self.history_series = df_all['history_series']
        self.horizon_series = df_all['horizon_series']
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx], self.history_series[idx], self.horizon_series[idx]



def main():
    parser = argparse.ArgumentParser(description="LLM Prediction with Examples")
    parser.add_argument("--csv_file", type=str, help="Path to the CSV file", default="")
    parser.add_argument("--llm_type", type=str, help="Choose the language language model", default='llama-3-8b')
    parser.add_argument("--hist_len", type=int, help="Historical time series length", default=96)
    parser.add_argument("--pred_len", type=int, help="Prediction length", default=12)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16)
    parser.add_argument("--start_iter", type=int, help="", default=0)
    parser.add_argument("--iter", type=int, help="Iteration", default=2)
    parser.add_argument("--save_dir", type=str, help="dir to save dataset", default='')
    parser.add_argument("--after_train", action='store_true', help="")
    parser.add_argument("--llm_path", type=str, help="", default="")
    args = parser.parse_args()

    tep = TextEhancementProvider(llm_name=args.llm_type, after_train=args.after_train, llm_path=args.llm_path)

    df_all = pd.read_csv(args.csv_file)
    data_set = temp_dataset(df_all)
    data_loader = DataLoader(data_set, args.batch_size, shuffle=False, num_workers=128, drop_last=False)

    accelerator = accelerate.Accelerator()
    tep, data_loader = accelerator.prepare(tep, data_loader)

    for iii in range(args.start_iter, args.iter):
        all_reinfoced_data = {
            'history_series': [],
            'horizon_series': [],
            'prompt': [],
            'reinforced_text': [],
        }
        for prompts, history_series, horizon_series in tqdm(data_loader, disable=not accelerator.is_local_main_process):
            reinforced_texts = tep.get_model_response(prompts)
            
            history_series = accelerator.gather_for_metrics(history_series)
            horizon_series = accelerator.gather_for_metrics(horizon_series)
            prompts = accelerator.gather_for_metrics(prompts)
            reinforced_texts = accelerator.gather_for_metrics(reinforced_texts)
            all_reinfoced_data['history_series'].extend(history_series)
            all_reinfoced_data['horizon_series'].extend(horizon_series)
            all_reinfoced_data['prompt'].extend(prompts)
            all_reinfoced_data['reinforced_text'].extend(reinforced_texts)

        if accelerator.is_local_main_process:
            os.makedirs(args.save_dir, exist_ok=True)
            all_reinfoced_data = pd.DataFrame(all_reinfoced_data)
            all_reinfoced_data.to_csv(os.path.join(args.save_dir, os.path.basename(args.csv_file))[:-4] + f'_{iii}.csv', index=False)

if __name__ == "__main__":
    main()