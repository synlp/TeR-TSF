import argparse
import pandas as pd
# from Modules.TEP import TextEhancementProvider
from Modules.MultimodalTSF import BaseMultimodalTSFModel
import ast
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import accelerate
import torch.distributed as dist
import torch
from make_dataset import TFHTSDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import warnings

warnings.simplefilter('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 必须在最开头设置！

def main():
    parser = argparse.ArgumentParser(description="LLM Prediction with Examples")
    parser.add_argument("--csv_file", type=str, help="Path to the CSV file", default="/media/ubuntu/data/collaborations/tsf/TeR-TSF/reinforced_my_datasets/Time-MMD/Agriculture/Agriculture_36_6_train_0.csv")
    parser.add_argument("--hist_len", type=int, help="Historical time series length", default=36)
    parser.add_argument("--pred_len", type=int, help="Prediction length", default=6)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    parser.add_argument("--save_dir", type=str, help="dir to save dataset", default='/media/ubuntu/data/collaborations/tsf/TeR-TSF/reward_my_datasets/Time-MMD/Agriculture/TextFusionHTS')
    parser.add_argument("--text_type", type=str, help="", default='reinforced_text')

    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    df_all = pd.read_csv(args.csv_file)
    data = args.csv_file.split("/")[-1].split("_")[0]
    df_all = df_all.dropna(axis=0, how='any', subset=[args.text_type])

    model_id = "/media/ubuntu/data/share/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    text_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    data_set = TFHTSDataset(df_all, args.hist_len, args.pred_len, args.text_type, tokenizer, text_model, data, device=device)
    data_loader = DataLoader(data_set, args.batch_size, shuffle=False, num_workers=128, drop_last=False)

    tsf_model = BaseMultimodalTSFModel(args.hist_len, args.pred_len, data, bm_name="textfusionhts")

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
    for batch_data in tqdm(data_loader):
        pred_serie = tsf_model.tfhts_predict_(batch_data["text_emb"], batch_data["ts"])

        # history_series = accelerator.gather_for_metrics(history_series)
        # horizon_series = accelerator.gather_for_metrics(horizon_series)
        # reinforced_texts = accelerator.gather_for_metrics(reinforced_texts)
        # pred_serie = accelerator.gather_for_metrics(pred_serie)
        
        all_data_with_reward['history_series'].extend(batch_data["ts"].cpu().numpy().tolist())
        all_data_with_reward['horizon_series'].extend(batch_data["pred"].cpu().numpy().tolist())
        all_data_with_reward['prompt'].extend(batch_data["prompt"])
        all_data_with_reward[args.text_type].extend(batch_data["text"])

        # pred_serie = np.concatenate(pred_serie).reshape(len(horizon_series), -1)
        reward = -np.mean((pred_serie.squeeze().cpu().numpy()- batch_data["pred"].cpu().numpy()) ** 2, axis=1)
        all_data_with_reward['reward1'].extend(reward.tolist())

    # if accelerator.is_local_main_process:
    os.makedirs(args.save_dir, exist_ok=True)
    all_data_with_reward = pd.DataFrame(all_data_with_reward)
    all_data_with_reward.to_csv(os.path.join(args.save_dir, os.path.basename(args.csv_file)), index=False)

    # dist.barrier()  # 确保所有进程都停止于此
    # dist.destroy_process_group()

if __name__ == "__main__":
    main()