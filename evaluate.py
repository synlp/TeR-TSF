

import argparse
import pandas as pd
import json
import torch
import gc
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
import random

from Modules.TeR import TextReinforcementModel
from utils.tools import (
    stat_dict, clear_dataloder, gen_text_dataset, 
    MCD_dataset, TFHTSDataset, Time_LLM_Dataset, 
    calculate_normalized_scores, RecordExpMetrics
)
from Modules.MultimodalTSF import BaseMultimodalTSFModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prepare_stage import gen_reinforced_text, get_data_paths, _init_tsf_model

warnings.filterwarnings('ignore')


SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_untrained_baseline_path(args, flag):
    
    base_path = os.path.join(args.data_dir, args.llm_type, args.tsf_type, args.data_name)
    
    reinforced_data_dir = os.path.join(base_path, "reinforced_data", flag, "untrained")
    baseline_path = os.path.join(
        reinforced_data_dir, 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
    )
    
    return baseline_path


def evaluate_prediction(args, tsf_model, flag="test", text_type="reinforced_text", trained=False):
  
    if text_type == "original_text":

        original_data_path = os.path.join(
            "/data2/user2/ter_tsf/processed_data", 
            f"{args.data_name}_{args.hist_len}_{args.pred_len}_{flag}.csv"
        )
        paths = get_data_paths(args, flag, trained=trained)
        paths['reinforced_data_file'] = original_data_path
    elif not trained and text_type == "reinforced_text":

        reinforced_data_file = get_untrained_baseline_path(args, flag)
        paths = get_data_paths(args, flag, trained=trained)
        paths['reinforced_data_file'] = reinforced_data_file
    else:

        paths = get_data_paths(args, flag, trained=trained)
    

    result_save_path = os.path.join(paths['base'], "results")
    os.makedirs(result_save_path, exist_ok=True)
    

    if text_type == "original_text":
        iter_idx = -1 
        record_text_type = "original_text"
    elif not trained:
        iter_idx = 0  
        record_text_type = "reinforced_text (untrained)"
    else:
        iter_idx = args.iter_idx
        record_text_type = "reinforced_text (trained)"
    
    fixed_params = {
        'Data': args.data_name, 'hist_len': args.hist_len, 'pred_len': args.pred_len
    }
    varying_params = {
        'iter': iter_idx, 'llm': args.llm_type, 'tsf': args.tsf_type, 
        'gen_num': args.gen_num, 'dpo_lr': args.dpo_lr, 'dpo_epoch': args.dpo_epoch, 
        'text': record_text_type
    }
    
    record_result = RecordExpMetrics(os.path.join(result_save_path, f"{flag}.json"))
    result_label = f"hist{args.hist_len}_pred{args.pred_len}_{args.llm_type}_{args.tsf_type}_gen{args.gen_num}_iter{iter_idx}_{text_type}_{trained}"
    
    if record_result.is_exist(fixed_params, varying_params):
        return
    
    if text_type == "original_text":
        df_all = pd.read_csv(paths['reinforced_data_file'])
       
    else:
        df_all = pd.read_csv(paths['reinforced_data_file'])
        df_all[[text_type]] = df_all[[text_type]].fillna("No text.")
    
    pred_dataset = _create_evaluation_dataset(args, df_all, text_type, tsf_model)
    pred_loader = DataLoader(pred_dataset, args.batch_size, shuffle=False, num_workers=128, drop_last=False)
    
    all_mse, all_mae = _execute_evaluation(args, tsf_model, pred_loader)
    
    mean_mse = np.mean(np.array(all_mse))
    mean_mae = np.mean(np.array(all_mae))
    
    result = {"MSE": mean_mse, "MAE": mean_mae}
    record_result.add_result(fixed_params, varying_params, result)
    
    print(f"eval done - MSE: {mean_mse:.6f}, MAE: {mean_mae:.6f}")
    
    del df_all, pred_dataset
    clear_dataloder(pred_loader)
    torch.cuda.empty_cache()
    gc.collect()


def _create_evaluation_dataset(args, df_all, text_type, tsf_model):
    model_id = "/data2/user2/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    text_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    return TFHTSDataset(df_all, args.hist_len, args.pred_len, text_type, tokenizer, text_model, args.data_name, device=tsf_model.device)


def _execute_evaluation(args, tsf_model, pred_loader):
    all_mse, all_mae = [], []
    
    for batch_data in tqdm(pred_loader, desc="执行评估"):
        pred_series = _get_prediction(args, tsf_model, batch_data)
    
        if batch_data['history_series'].size(0) == 1:
            horizon_series = batch_data['horizon_series'].squeeze().unsqueeze(dim=0)
            pred_series = pred_series.squeeze().unsqueeze(dim=0)
        else:
            horizon_series = batch_data['horizon_series'].squeeze()
            pred_series = pred_series.squeeze()
        
        mse = np.mean((pred_series.cpu().numpy() - horizon_series.cpu().numpy()) ** 2, axis=1)
        mae = np.mean(np.abs(pred_series.cpu().numpy() - horizon_series.cpu().numpy()), axis=1)
        
        all_mse.extend(mse.tolist())
        all_mae.extend(mae.tolist())
    
    return all_mse, all_mae


def _get_prediction(args, tsf_model, batch_data):
    return tsf_model.tfhts_predict_(batch_data["text_emb"], batch_data["history_series"]).squeeze()


def main():
    parser = argparse.ArgumentParser(description="TeR-TSF eval")
    parser.add_argument("--data_dir", type=str, default="/data2/user2/ter_tsf")
    parser.add_argument("--data_name", type=str, default="Agriculture")
    parser.add_argument("--llm_type", type=str, default='qwen3-1.7b')
    parser.add_argument("--tsf_type", type=str, default='tfhts')
    parser.add_argument("--hist_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iter_idx", type=int, default=0)
    parser.add_argument("--llm_path", type=str, default="original")
    parser.add_argument("--exp_time", type=str, default="")
    parser.add_argument("--gen_num", type=int, default=2,)
    parser.add_argument("--down_sample", type=int, default=1)
    parser.add_argument("--dpo_epoch", type=int, default=5)
    parser.add_argument("--dpo_lr", type=float, default=0.0001)
    args = parser.parse_args()

    
    tep = TextReinforcementModel(llm_name=args.llm_type, llm_path=args.llm_path)
    
    tsf_model = _init_tsf_model(args)
    

    evaluate_prediction(args, tsf_model, flag="test", text_type="original_text")

    if args.iter_idx == 0:
        evaluate_prediction(args, tsf_model, flag="test", text_type="reinforced_text", trained=False)
    else:
        print("Skip...")

    gen_reinforced_text(args, tep, flag="test", trained=True)
    
    evaluate_prediction(args, tsf_model, flag="test", text_type="reinforced_text", trained=True)
    
    del tep
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()