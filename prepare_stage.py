

import argparse
import pandas as pd
import json
import torch
import gc
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
import random

from Modules.TeR import TextReinforcementModel
from utils.tools import (
    stat_dict, clear_dataloder, gen_text_dataset, 
    MCD_dataset, TFHTSDataset, Time_LLM_Dataset, 
    calculate_normalized_scores
)
from Modules.MultimodalTSF import BaseMultimodalTSFModel
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_data_paths(args, flag="train", gen_idx=None, trained=False):

    base_path = os.path.join(args.data_dir, args.llm_type, args.tsf_type, args.data_name)
    
    paths = {
        'base': base_path,
        'reinforced_data': os.path.join(base_path, "reinforced_data"),
        'reward_data': os.path.join(base_path, "reward_data"),
        'preference_data': os.path.join(base_path, "preference_data"),
        'original_performance': os.path.join(base_path, "original_performance")
    }
    

    paths['original_data'] = os.path.join(
        "/data2/user2/ter_tsf/processed_data", 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_{flag}.csv"
    )
    

    if flag == "train":

        paths['reinforced_data_dir'] = os.path.join(paths['reinforced_data'], flag, f"iter{args.iter_idx}")
        paths['reinforced_data_file'] = os.path.join(
            paths['reinforced_data_dir'], 
            f"{args.data_name}_{args.hist_len}_{args.pred_len}_gen{gen_idx}_{args.exp_time}.csv"
        )
    else:

        if not trained:

            paths['reinforced_data_dir'] = os.path.join(paths['reinforced_data'], flag, "untrained")
            paths['reinforced_data_file'] = os.path.join(
                paths['reinforced_data_dir'], 
                f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
            )
        else:

            paths['reinforced_data_dir'] = os.path.join(paths['reinforced_data'], flag, f"iter{args.iter_idx}")
            paths['reinforced_data_file'] = os.path.join(
                paths['reinforced_data_dir'], 
                f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
            )
    
    
    if flag == "train":
        
        paths['reward_data_dir'] = os.path.join(paths['reward_data'], flag, f"iter{args.iter_idx}")
        paths['reward_data_file'] = os.path.join(
            paths['reward_data_dir'], 
            f"{args.data_name}_{args.hist_len}_{args.pred_len}_gen{gen_idx}_{args.exp_time}.csv"
        )
    else:

        if not trained:

            paths['reward_data_dir'] = os.path.join(paths['reward_data'], flag, "untrained")
            paths['reward_data_file'] = os.path.join(
                paths['reward_data_dir'], 
                f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
            )
        else:

            paths['reward_data_dir'] = os.path.join(paths['reward_data'], flag, f"iter{args.iter_idx}")
            paths['reward_data_file'] = os.path.join(
                paths['reward_data_dir'], 
                f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
            )
    
    return paths


def evaluate_original_performance(args, tsf_model):
    
    paths = get_data_paths(args, "train")
    os.makedirs(paths['original_performance'], exist_ok=True)
    original_perf_path = os.path.join(
        paths['original_performance'], 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.tsf_type}_train_mse.csv"
    )
    

    if os.path.isfile(original_perf_path):
        return pd.read_csv(original_perf_path)

    
    df_all = pd.read_csv(paths['original_data'])

    pred_dataset = _create_prediction_dataset(args, df_all, "original_text", tsf_model)
    pred_loader = DataLoader(pred_dataset, args.batch_size, shuffle=False, num_workers=128, drop_last=False)
    

    all_mse = []
    history_series_list = []
    
    for batch_data in tqdm(pred_loader):

        pred_series = _get_prediction(args, tsf_model, batch_data)
        

        mse = np.mean((pred_series.squeeze().cpu().numpy() - batch_data['horizon_series'].squeeze().cpu().numpy()) ** 2, axis=1)
        all_mse.extend(mse.tolist())
        

        if batch_data['history_series'].size(0) == 1:
            history_series = batch_data['history_series'].squeeze().unsqueeze(dim=0)
        else:
            history_series = batch_data['history_series'].squeeze()
        history_series_list.extend(history_series.cpu().numpy().tolist())
    

    original_perf_df = pd.DataFrame({
        'history_series': history_series_list,
        'original_mse': all_mse
    })
    
    original_perf_df.to_csv(original_perf_path, index=False)
    
    del df_all, pred_dataset
    clear_dataloder(pred_loader)
    torch.cuda.empty_cache()
    gc.collect()
    
    return original_perf_df


def gen_reinforced_text(args, tep, flag="train", trained=False):

    paths = get_data_paths(args, flag, trained=trained)
    
    df_all = pd.read_csv(paths['original_data'])
    dataset = gen_text_dataset(df_all, down_sample=args.down_sample)
    data_loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=128, drop_last=False)
    
    if flag == "train":
        for gen_idx in range(args.gen_num):
            gen_paths = get_data_paths(args, flag, gen_idx, trained)

            os.makedirs(gen_paths['reinforced_data_dir'], exist_ok=True)
            
            if os.path.isfile(gen_paths['reinforced_data_file']):
                continue
                
            _generate_and_save_texts(data_loader, tep, gen_paths['reinforced_data_file'], args.max_batches)
    else:
        os.makedirs(paths['reinforced_data_dir'], exist_ok=True)
        
        if not os.path.isfile(paths['reinforced_data_file']):
            _generate_and_save_texts(data_loader, tep, paths['reinforced_data_file'], args.max_batches)
        else:
            print(f"Skip: {paths['reinforced_data_file']}")
    
    # 清理内存
    del df_all
    clear_dataloder(data_loader)
    torch.cuda.empty_cache()
    gc.collect()


def _generate_and_save_texts(data_loader, tep, save_path, max_batches=None):
    all_data = {
        'history_series': [], 'horizon_series': [], 
        'prompt': [], 'reinforced_text': []
    }
    
    batch_count = 0
    
    for prompts, history_series, horizon_series in tqdm(data_loader):
        reinforced_texts = tep.get_model_response(prompts)
        
        all_data['history_series'].extend(history_series)
        all_data['horizon_series'].extend(horizon_series)
        all_data['prompt'].extend(prompts)
        all_data['reinforced_text'].extend(reinforced_texts)
        
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break
    
    pd.DataFrame(all_data).to_csv(save_path, index=False)


def multimodal_prediction(args, tsf_model, flag="train", text_type="reinforced_text", gen_idx=0):

    paths = get_data_paths(args, flag, gen_idx)
    

    os.makedirs(paths['reward_data_dir'], exist_ok=True)
    
    if os.path.isfile(paths['reward_data_file']):
        return
    

    df_all = pd.read_csv(paths['reinforced_data_file'])
    df_all = df_all.dropna(axis=0, how='any', subset=[text_type])
    df_all.reset_index(drop=True, inplace=True)

    pred_dataset = _create_prediction_dataset(args, df_all, text_type, tsf_model)
    pred_loader = DataLoader(pred_dataset, args.batch_size, shuffle=False, num_workers=128, drop_last=False)
   
    all_data_with_reward = _execute_prediction_and_reward(args, tsf_model, pred_loader, text_type)
    

    pd.DataFrame(all_data_with_reward).to_csv(paths['reward_data_file'], index=False)

    del df_all, all_data_with_reward, pred_dataset
    clear_dataloder(pred_loader)
    torch.cuda.empty_cache()
    gc.collect()


def _create_prediction_dataset(args, df_all, text_type, tsf_model):

    model_id = "/data2/user2/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    text_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    return TFHTSDataset(df_all, args.hist_len, args.pred_len, text_type, tokenizer, text_model, args.data_name, device=tsf_model.device)


def _execute_prediction_and_reward(args, tsf_model, pred_loader, text_type):
    all_data_with_reward = {
        'history_series': [], 'horizon_series': [], 
        'prompt': [], text_type: [], 'reward1': []
    }
    
    for batch_data in tqdm(pred_loader):

        pred_series = _get_prediction(args, tsf_model, batch_data)
        
        if batch_data['history_series'].size(0) == 1:
            history_series = batch_data['history_series'].squeeze().unsqueeze(dim=0)
            horizon_series = batch_data['horizon_series'].squeeze().unsqueeze(dim=0)
            pred_series = pred_series.squeeze().unsqueeze(dim=0)
            all_data_with_reward['history_series'].extend(history_series.cpu().numpy().tolist())
            all_data_with_reward['horizon_series'].extend(horizon_series.cpu().numpy().tolist())
            all_data_with_reward['prompt'].append(batch_data['prompt'])
            all_data_with_reward[text_type].append(batch_data[text_type])
        else:
            history_series = batch_data['history_series'].squeeze()
            horizon_series = batch_data['horizon_series'].squeeze()
            pred_series = pred_series.squeeze()
            all_data_with_reward['history_series'].extend(history_series.cpu().numpy().tolist())
            all_data_with_reward['horizon_series'].extend(horizon_series.cpu().numpy().tolist())
            all_data_with_reward['prompt'].extend(batch_data['prompt'])
            all_data_with_reward[text_type].extend(batch_data[text_type])
        
        reward = -np.mean((pred_series.cpu().numpy() - horizon_series.cpu().numpy()) ** 2, axis=1)
        all_data_with_reward['reward1'].extend(reward.tolist())
    
    return all_data_with_reward


def _get_prediction(args, tsf_model, batch_data):

    return tsf_model.tfhts_predict_(batch_data["text_emb"], batch_data["history_series"]).squeeze()


def get_preference_data(args, original_perf_df):

    paths = get_data_paths(args)
    os.makedirs(paths['preference_data'], exist_ok=True)
    preference_path = os.path.join(
        paths['preference_data'], 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_genNum{args.gen_num}_iter{args.iter_idx}_{args.exp_time}.json"
    )
    
    if os.path.isfile(preference_path):
        return
    

    df1 = _load_reward_data(args, 0)
    for gen_idx in range(1, args.gen_num):
        df2 = _load_reward_data(args, gen_idx)
        df1 = _merge_reward_data(df1, df2, gen_idx)
    

    prev_perf_df = get_previous_iteration_performance(args)
    
    if prev_perf_df is not None:
        if 'history_series_str' not in df1.columns:
            df1['history_series_str'] = df1['history_series'].astype(str)
        if 'history_series_str' not in prev_perf_df.columns:
            prev_perf_df['history_series_str'] = prev_perf_df['history_series'].astype(str)
        df1 = pd.merge(df1, prev_perf_df[['history_series_str', 'mse']], 
                       on='history_series_str', how='left')
        df1 = df1.rename(columns={'mse': 'prev_mse'})
    
    df1 = _filter_effective_texts(df1, original_perf_df, args.gen_num, args)

    if len(df1) == 0:
        sys.exit(1) 
    
    total_samples = len(_load_reward_data(args, 0))
    effective_ratio = len(df1) / total_samples
    
    if not args.disable_text_quality_reward:
        for gen_idx in range(args.gen_num):
            total_quality_score = 0
            for row_idx in range(len(df1)):
                score = calculate_normalized_scores(df1.loc[row_idx, f"reinforced_text_{gen_idx}"])
                df1.loc[row_idx, f"reward_{gen_idx}"] += args.lam * score
                total_quality_score += score
            
    else:
        print("Skip")
    
    
    preference_dataset = _generate_preference_pairs(df1, args.gen_num)
    
    with open(preference_path, 'w') as f:
        json.dump(preference_dataset, f, indent=4)
    
    _update_dataset_info(args, preference_path)
    


def _filter_effective_texts(df1, original_perf_df, gen_num, args):
    
    df1['history_series_str'] = df1['history_series'].astype(str)
    original_perf_df['history_series_str'] = original_perf_df['history_series'].astype(str)
    
    df1 = pd.merge(df1, original_perf_df[['history_series_str', 'original_mse']], 
                   on='history_series_str', how='left')
    
    prev_perf_df = get_previous_iteration_performance(args)
    
    if prev_perf_df is not None:
        if 'history_series_str' not in df1.columns:
            df1['history_series_str'] = df1['history_series'].astype(str)
        if 'history_series_str' not in prev_perf_df.columns:
            prev_perf_df['history_series_str'] = prev_perf_df['history_series'].astype(str)
        df1 = pd.merge(df1, prev_perf_df[['history_series_str', 'mse']], 
                       on='history_series_str', how='left')
        df1 = df1.rename(columns={'mse': 'prev_mse'})
    
    effective_mask = pd.Series([False] * len(df1))
    performance_improvement_count = 0
    acceptable_degradation_count = 0
    iteration_improvement_count = 0
    
    for idx in range(len(df1)):
        sample_effective = False
        best_improvement_vs_original = float('-inf')
        best_improvement_vs_prev = float('-inf')
        
        original_mse = df1.loc[idx, 'original_mse']
        prev_mse = df1.loc[idx, 'prev_mse'] if prev_perf_df is not None else None
        
        if hasattr(prev_mse, 'iloc'):
            prev_mse = prev_mse.iloc[0]

        for gen_idx in range(gen_num):
            generated_mse = -df1.loc[idx, f'reward_{gen_idx}']

            improvement_vs_original = (original_mse - generated_mse) / original_mse
            best_improvement_vs_original = max(best_improvement_vs_original, improvement_vs_original)
            
            if prev_mse is not None:
                improvement_vs_prev = (prev_mse - generated_mse) / prev_mse
                best_improvement_vs_prev = max(best_improvement_vs_prev, improvement_vs_prev)
            
            
            condition_original = improvement_vs_original > -float('inf')
            condition_prev = True
            
            if prev_mse is not None:
                condition_prev = improvement_vs_prev > -float('inf')
            
            if condition_original and condition_prev:
                sample_effective = True
                
                if improvement_vs_original > 0.02:
                    performance_improvement_count += 1
                elif improvement_vs_original > -0.50:
                    acceptable_degradation_count += 1
                
                if prev_mse is not None and improvement_vs_prev > 0.01:
                    iteration_improvement_count += 1
                
                break
        
        effective_mask[idx] = sample_effective
        
    
    df1_filtered = df1[effective_mask].copy()
    
    columns_to_drop = ['history_series_str', 'original_mse']
    if prev_perf_df is not None:
        columns_to_drop.append('prev_mse')
    df1_filtered = df1_filtered.drop(columns_to_drop, axis=1)
    
    df1_filtered = df1_filtered.reset_index(drop=True)
    
    
    return df1_filtered


def _load_reward_data(args, gen_idx):
    paths = get_data_paths(args, "train", gen_idx)
    df = pd.read_csv(paths['reward_data_file'])
    df[f'reward_{gen_idx}'] = df.iloc[:, 4:].sum(axis=1)
    df = df.reset_index(drop=True)
    result_df = df[['history_series', 'prompt', 'reinforced_text', f'reward_{gen_idx}']].copy()
    result_df = result_df.rename(columns={'reinforced_text': f'reinforced_text_{gen_idx}'})
    return result_df


def _merge_reward_data(df1, df2, gen_idx):
    merged_df = pd.merge(df1, df2, how='inner', on='history_series')
    
    if 'prompt_x' in merged_df.columns and 'prompt_y' in merged_df.columns:
        merged_df['prompt'] = merged_df['prompt_x']
        merged_df = merged_df.drop(['prompt_x', 'prompt_y'], axis=1)
    elif 'prompt_x' in merged_df.columns:
        merged_df['prompt'] = merged_df['prompt_x']
        merged_df = merged_df.drop('prompt_x', axis=1)
    elif 'prompt_y' in merged_df.columns:
        merged_df['prompt'] = merged_df['prompt_y']
        merged_df = merged_df.drop('prompt_y', axis=1)
    
    return merged_df.reset_index(drop=True)


def _generate_preference_pairs(df1, gen_num):
    reward_cols = [c for c in df1.columns if 'reward' in c]
    reward_data = df1[reward_cols]
    

    pos = reward_data.idxmax(axis=1).apply(lambda x: 'reinforced_text'+x[-2:])
    neg = reward_data.idxmin(axis=1).apply(lambda x: 'reinforced_text'+x[-2:])
    

    preference_dataset = []
    for idx in range(len(pos)):
        
        preference_dataset.append({
            "conversations": [{"from": "human", "value": df1.loc[idx, 'prompt']}],
            "chosen": {"from": "gpt", "value": df1.loc[idx, pos[idx]]},
            "rejected": {"from": "gpt", "value": df1.loc[idx, neg[idx]]}
        })
    
    return preference_dataset


def _update_dataset_info(args, preference_path):
    dataset_name = f"{args.data_name}_h{args.hist_len}_p{args.pred_len}_{args.llm_type}_{args.tsf_type}_genNum{args.gen_num}_iter{args.iter_idx}_{args.exp_time}"
    
    new_data_info = {
        dataset_name: {
            "file_name": preference_path,
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    }
    
    dataset_info_path = os.path.join(args.llama_factory_dir, "data/dataset_info.json")
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        data_info = json.load(f)
        data_info.update(new_data_info)
    
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(data_info, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="TeR-TSF")
    parser.add_argument("--data_dir", type=str, default="/data2/user2/ter_tsf")
    parser.add_argument("--data_name", type=str, default="Agriculture")
    parser.add_argument("--llm_type", type=str, default='qwen3-1.7b')
    parser.add_argument("--tsf_type", type=str, default='mcd_tsf')
    parser.add_argument("--hist_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gen_num", type=int, default=2)
    parser.add_argument("--iter_idx", type=int, default=0)
    parser.add_argument("--llm_path", type=str, default="original")
    parser.add_argument("--exp_time", type=str, default="")
    parser.add_argument("--down_sample", type=int, default=0)
    parser.add_argument("--llama_factory_dir", type=str, default="")
    parser.add_argument("--lam", type=float, default=1.5)
    parser.add_argument("--disable_text_quality_reward", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()
    
    print("================================")
    
    original_data_path = os.path.join(
        "./", 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_train.csv"
    )
    if not os.path.exists(original_data_path):
        sys.exit(2) 
    
    try:
        tep = TextReinforcementModel(llm_name=args.llm_type, llm_path=args.llm_path)
    except Exception as e:
        sys.exit(3)  
    
    try:
        tsf_model = _init_tsf_model(args)
    except Exception as e:
        sys.exit(4) 
    
    try:
        original_perf_df = evaluate_original_performance(args, tsf_model)
    except Exception as e:
        sys.exit(5)
    
    gen_reinforced_text(args, tep, flag="train")
    if args.iter_idx == 0:
        gen_reinforced_text(args, tep, flag="val", trained=False)
        gen_reinforced_text(args, tep, flag="test", trained=False)
    else:
        gen_reinforced_text(args, tep, flag="val", trained=True)
        gen_reinforced_text(args, tep, flag="test", trained=True)
    
    del tep
    torch.cuda.empty_cache()
    gc.collect()
    
    for gen_idx in range(args.gen_num):
        multimodal_prediction(args, tsf_model, flag="train", text_type="reinforced_text", gen_idx=gen_idx)

    get_preference_data(args, original_perf_df)
    
    gen0_reward_data = _load_reward_data(args, 0)
    record_iteration_performance(args, gen0_reward_data, gen_idx=0)
    
    analyze_training_progress(args)
    


def _init_tsf_model(args):

    return BaseMultimodalTSFModel(args.hist_len, args.pred_len, args.data_name, bm_name="textfusionhts", llm_type=args.llm_type, iter_=args.iter_idx, exp_time=args.exp_time)


def record_iteration_performance(args, df1, gen_idx=1):

    perf_record_dir = os.path.join(
        args.data_dir, args.llm_type, args.tsf_type, args.data_name,
        "iteration_performance", f"{args.hist_len}_{args.pred_len}_{args.exp_time}"
    )
    os.makedirs(perf_record_dir, exist_ok=True)
    
    perf_record_path = os.path.join(
        perf_record_dir,
        f"iter{args.iter_idx}_performance.csv"
    )
    
    reward_col = f'reward_{gen_idx}'
    
    perf_data = df1[['history_series', reward_col]].copy()
    perf_data['mse'] = -perf_data[reward_col] 
    perf_data['iter_idx'] = args.iter_idx
    perf_data['gen_idx'] = gen_idx
    
    perf_data['history_series_str'] = perf_data['history_series'].astype(str)
    
    perf_data.to_csv(perf_record_path, index=False)
    
    mean_mse = perf_data['mse'].mean()
    std_mse = perf_data['mse'].std()
    
    return perf_data


def get_previous_iteration_performance(args):

    if args.iter_idx == 0:
        return None
    
    perf_record_dir = os.path.join(
        args.data_dir, args.llm_type, args.tsf_type, args.data_name,
        "iteration_performance", f"{args.hist_len}_{args.pred_len}_{args.exp_time}"
    )
    
    prev_perf_path = os.path.join(
        perf_record_dir,
        f"iter{args.iter_idx-1}_performance.csv"
    )
    
    if os.path.exists(prev_perf_path):
        try:
            prev_perf_df = pd.read_csv(prev_perf_path)
            
            if 'history_series_str' not in prev_perf_df.columns:
                prev_perf_df['history_series_str'] = prev_perf_df['history_series'].astype(str)
            return prev_perf_df
        except Exception as e:
            return None
    else:
        return None


def analyze_training_progress(args):

    perf_record_dir = os.path.join(
        args.data_dir, args.llm_type, args.tsf_type, args.data_name,
        "iteration_performance", f"{args.hist_len}_{args.pred_len}_{args.exp_time}"
    )
    
    if not os.path.exists(perf_record_dir):
        return
    
    all_perf_data = []
    for iter_idx in range(args.iter_idx + 1):
        perf_path = os.path.join(perf_record_dir, f"iter{iter_idx}_performance.csv")
        if os.path.exists(perf_path):
            perf_df = pd.read_csv(perf_path)
            perf_df['iter_idx'] = iter_idx
            all_perf_data.append(perf_df)
    
    if not all_perf_data:
        return
    
    combined_perf = pd.concat(all_perf_data, ignore_index=True)
    

    iter_summary = combined_perf.groupby('iter_idx')['mse'].agg(['mean', 'std', 'count']).reset_index()
    
    
    for _, row in iter_summary.iterrows():
        print(f"iter{row['iter_idx']}\t{row['mean']:.6f}\t{row['std']:.6f}\t{row['count']}")
    

    if len(iter_summary) > 1:
        for i in range(1, len(iter_summary)):
            prev_mse = iter_summary.iloc[i-1]['mean']
            curr_mse = iter_summary.iloc[i]['mean']
            improvement = (prev_mse - curr_mse) / prev_mse * 100
            
    
    original_perf_path = os.path.join(
        args.data_dir, args.llm_type, args.tsf_type, args.data_name,
        "original_performance", 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.tsf_type}_train_mse.csv"
    )
    
    if os.path.exists(original_perf_path):
        original_perf = pd.read_csv(original_perf_path)
        original_mean_mse = original_perf['original_mse'].mean()
        


if __name__ == "__main__":
    main()