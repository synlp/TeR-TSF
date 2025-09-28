"""
TeR-TSF 数据准备阶段脚本

逻辑流程图：
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   原始数据       │    │   文本增强生成    │    │   多模态预测     │
│ (processed_data) │───▶│ (reinforced_data) │───▶│  (reward_data)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   性能筛选        │    │   偏好数据生成   │
                       │(性能对比筛选)     │    │(有效文本偏好对)  │
                       └──────────────────┘    └─────────────────┘

文件路径结构：
原始数据路径：
/data2/user2/ter_tsf/processed_data/
├── {data_name}_{hist_len}_{pred_len}_train.csv
├── {data_name}_{hist_len}_{pred_len}_val.csv
└── {data_name}_{hist_len}_{pred_len}_test.csv

生成数据路径：
/data2/user2/ter_tsf/
├── {llm_type}/                    # 按LLM类型分类
│   ├── {tsf_type}/               # 按TSF类型分类
│   │   ├── {data_name}/          # 按数据集分类
│   │   │   ├── reinforced_data/  # 增强文本数据
│   │   │   │   ├── train/        # 训练集
│   │   │   │   │   ├── iter0/    # 第0轮训练
│   │   │   │   │   │   ├── gen0.csv  # 第0轮第0次生成
│   │   │   │   │   │   ├── gen1.csv  # 第0轮第1次生成
│   │   │   │   │   │   └── ...       # 第0轮第(gen_num-1)次生成
│   │   │   │   │   ├── iter1/    # 第1轮训练
│   │   │   │   │   │   ├── gen0.csv  # 第1轮第0次生成
│   │   │   │   │   │   ├── gen1.csv  # 第1轮第1次生成
│   │   │   │   │   │   └── ...       # 第1轮第(gen_num-1)次生成
│   │   │   │   │   └── ...
│   │   │   │   ├── val/          # 验证集
│   │   │   │   │   ├── untrained.csv  # 未训练模型生成
│   │   │   │   │   └── iter{i}.csv    # 第i轮训练后生成
│   │   │   │   └── test/         # 测试集
│   │   │   │       ├── untrained.csv  # 未训练模型生成
│   │   │   │       └── iter{i}.csv    # 第i轮训练后生成
│   │   │   ├── reward_data/      # 奖励数据
│   │   │   │   ├── train/
│   │   │   │   │   ├── iter0/
│   │   │   │   │   │   ├── gen0.csv  # 第0轮第0次生成的奖励
│   │   │   │   │   │   ├── gen1.csv  # 第0轮第1次生成的奖励
│   │   │   │   │   │   └── ...       # 第0轮第(gen_num-1)次生成的奖励
│   │   │   │   │   ├── iter1/
│   │   │   │   │   │   ├── gen0.csv  # 第1轮第0次生成的奖励
│   │   │   │   │   │   ├── gen1.csv  # 第1轮第1次生成的奖励
│   │   │   │   │   │   └── ...       # 第1轮第(gen_num-1)次生成的奖励
│   │   │   │   │   └── ...
│   │   │   │   ├── val/
│   │   │   │   │   ├── untrained.csv  # 未训练模型生成的奖励
│   │   │   │   │   └── iter{i}.csv    # 第i轮训练后生成的奖励
│   │   │   │   └── test/
│   │   │   │       ├── untrained.csv  # 未训练模型生成的奖励
│   │   │   │       └── iter{i}.csv    # 第i轮训练后生成的奖励
│   │   │   ├── preference_data/  # 偏好数据
│   │   │   │   └── iter{i}_gen{gen_num}.json
│   │   │   └── original_performance/  # 原始性能数据
│   │   │       └── {data_name}_{hist_len}_{pred_len}_{tsf_type}_train_mse.csv
│   │   └── ...
│   └── ...
└── ...

设计逻辑说明：
1. 训练集：每个轮次独立生成gen_num个文本，用于DPO训练时的偏好对比较
   - gen0: 当前轮第0次生成的文本
   - gen1: 当前轮第1次生成的文本
   - ...
   - gen{gen_num-1}: 当前轮第(gen_num-1)次生成的文本

2. 验证集/测试集：
   - untrained/: 未训练模型生成的文本（固定基准线）
   - iter{i}/: 第i轮训练后模型生成的文本（用于评测）

3. 偏好对生成：
   - 比较同一轮次内不同生成次数的文本质量
   - 选择奖励最高的作为chosen，最低的作为rejected

主要功能：
1. 使用LLM生成增强文本描述
2. 结合时间序列数据进行多模态预测
3. 计算预测奖励并生成偏好数据
4. 评估原始性能并筛选有效文本
5. 更新LLaMA-Factory数据集配置

输入：原始时间序列数据（包含original_text列）+ 提示词
输出：增强文本 + 偏好数据集 + 奖励数据
"""

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
    """
    获取数据文件路径
    
    Args:
        args: 参数配置
        flag: 数据集类型 ("train"/"val"/"test")
        gen_idx: 生成索引（仅训练集需要，表示当前轮第gen_idx次生成）
        trained: 是否使用训练后的模型（仅验证集/测试集需要）
    
    Returns:
        dict: 包含各种路径的字典
    
    路径逻辑说明：
    1. 训练集 (flag="train"):
       - 每个轮次独立生成gen_num个文本（gen_num可以是任意正整数）
       - gen_idx表示当前轮第gen_idx次生成的文本（0 ≤ gen_idx < gen_num）
       - 路径: iter{iter_idx}/gen{gen_idx}_{exp_time}.csv
    
    2. 验证集/测试集 (flag="val"/"test"):
       - trained=False: 使用未训练模型生成的文本
       - trained=True: 使用当前轮训练后模型生成的文本
       - 路径: untrained/{data}_{hist}_{pred}_{exp}.csv 或 iter{iter_idx}/{data}_{hist}_{pred}_{exp}.csv
    """
    # 基础路径结构：{data_dir}/{llm_type}/{tsf_type}/{data_name}/
    base_path = os.path.join(args.data_dir, args.llm_type, args.tsf_type, args.data_name)
    
    paths = {
        'base': base_path,
        'reinforced_data': os.path.join(base_path, "reinforced_data"),
        'reward_data': os.path.join(base_path, "reward_data"),
        'preference_data': os.path.join(base_path, "preference_data"),
        'original_performance': os.path.join(base_path, "original_performance")
    }
    
    # 原始数据路径
    paths['original_data'] = os.path.join(
        "/data2/user2/ter_tsf/processed_data", 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_{flag}.csv"
    )
    
    # 增强文本数据路径
    if flag == "train":
        # 训练集：每个轮次独立生成gen_num个文本用于偏好对比较
        # gen_idx表示当前轮第gen_idx次生成的文本
        paths['reinforced_data_dir'] = os.path.join(paths['reinforced_data'], flag, f"iter{args.iter_idx}")
        paths['reinforced_data_file'] = os.path.join(
            paths['reinforced_data_dir'], 
            f"{args.data_name}_{args.hist_len}_{args.pred_len}_gen{gen_idx}_{args.exp_time}.csv"
        )
    else:
        # 验证集和测试集：区分未训练和训练后的情况
        if not trained:
            # 未训练情况：使用untrained目录
            paths['reinforced_data_dir'] = os.path.join(paths['reinforced_data'], flag, "untrained")
            paths['reinforced_data_file'] = os.path.join(
                paths['reinforced_data_dir'], 
                f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
            )
        else:
            # 训练后情况：使用iter{i}目录
            paths['reinforced_data_dir'] = os.path.join(paths['reinforced_data'], flag, f"iter{args.iter_idx}")
            paths['reinforced_data_file'] = os.path.join(
                paths['reinforced_data_dir'], 
                f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
            )
    
    # 奖励数据路径
    if flag == "train":
        # 训练集：每个轮次独立生成gen_num个奖励数据（gen_num可以是任意正整数）
        # 对应增强文本的奖励计算
        paths['reward_data_dir'] = os.path.join(paths['reward_data'], flag, f"iter{args.iter_idx}")
        paths['reward_data_file'] = os.path.join(
            paths['reward_data_dir'], 
            f"{args.data_name}_{args.hist_len}_{args.pred_len}_gen{gen_idx}_{args.exp_time}.csv"
        )
    else:
        # 验证集和测试集：对应增强文本的奖励计算
        if not trained:
            # 未训练情况：使用untrained目录
            paths['reward_data_dir'] = os.path.join(paths['reward_data'], flag, "untrained")
            paths['reward_data_file'] = os.path.join(
                paths['reward_data_dir'], 
                f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
            )
        else:
            # 训练后情况：使用iter{i}目录
            paths['reward_data_dir'] = os.path.join(paths['reward_data'], flag, f"iter{args.iter_idx}")
            paths['reward_data_file'] = os.path.join(
                paths['reward_data_dir'], 
                f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
            )
    
    return paths


def evaluate_original_performance(args, tsf_model):
    """
    评估多模态预测模型在原始数据上的性能
    
    Args:
        args: 参数配置
        tsf_model: 时间序列预测模型
    
    Returns:
        pd.DataFrame: 包含原始MSE的数据框
    """
    # 获取路径
    paths = get_data_paths(args, "train")
    os.makedirs(paths['original_performance'], exist_ok=True)
    original_perf_path = os.path.join(
        paths['original_performance'], 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.tsf_type}_train_mse.csv"
    )
    
    # 检查是否已存在原始性能数据
    if os.path.isfile(original_perf_path):
        print(f"原始性能数据已存在，加载: {original_perf_path}")
        return pd.read_csv(original_perf_path)
    
    print("开始评估原始数据性能...")
    
    # 加载原始数据
    df_all = pd.read_csv(paths['original_data'])
    # 原始数据中已包含original_text列，无需添加占位符
    
    # 创建数据集
    pred_dataset = _create_prediction_dataset(args, df_all, "original_text", tsf_model)
    pred_loader = DataLoader(pred_dataset, args.batch_size, shuffle=False, num_workers=128, drop_last=False)
    
    # 执行预测并计算MSE
    all_mse = []
    history_series_list = []
    
    for batch_data in tqdm(pred_loader, desc="评估原始性能"):
        # 执行预测
        pred_series = _get_prediction(args, tsf_model, batch_data)
        
        # 计算MSE
        mse = np.mean((pred_series.squeeze().cpu().numpy() - batch_data['horizon_series'].squeeze().cpu().numpy()) ** 2, axis=1)
        all_mse.extend(mse.tolist())
        
        # 保存history_series用于后续匹配
        if batch_data['history_series'].size(0) == 1:
            history_series = batch_data['history_series'].squeeze().unsqueeze(dim=0)
        else:
            history_series = batch_data['history_series'].squeeze()
        history_series_list.extend(history_series.cpu().numpy().tolist())
    
    # 创建结果数据框
    original_perf_df = pd.DataFrame({
        'history_series': history_series_list,
        'original_mse': all_mse
    })
    
    # 保存原始性能数据
    original_perf_df.to_csv(original_perf_path, index=False)
    print(f"原始性能数据已保存: {original_perf_path}")
    
    # 清理内存
    del df_all, pred_dataset
    clear_dataloder(pred_loader)
    torch.cuda.empty_cache()
    gc.collect()
    
    return original_perf_df


def gen_reinforced_text(args, tep, flag="train", trained=False):
    """
    生成增强文本数据
    
    Args:
        args: 参数配置
        tep: 文本增强模型
        flag: 数据集类型 ("train"/"val"/"test")
        trained: 是否使用训练后的模型（仅验证集/测试集需要）
    
    生成逻辑：
    1. 训练集：每个轮次生成gen_num个文本（gen_num可以是任意正整数），用于DPO训练时的偏好对比较
    2. 验证集/测试集：生成单个文本，用于评测
       - trained=False: 使用未训练模型生成（固定基准线）
       - trained=True: 使用当前轮训练后模型生成
    """
    # 获取路径
    paths = get_data_paths(args, flag, trained=trained)
    
    # 加载原始数据
    df_all = pd.read_csv(paths['original_data'])
    dataset = gen_text_dataset(df_all, down_sample=args.down_sample)
    data_loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=128, drop_last=False)
    
    # 确定保存路径
    if flag == "train":
        # 训练集：生成多份数据用于偏好对比较
        # 每个轮次独立生成gen_num个文本（gen_num可以是任意正整数）
        for gen_idx in range(args.gen_num):
            # 获取当前生成索引的路径
            gen_paths = get_data_paths(args, flag, gen_idx, trained)
            
            # 创建目录
            os.makedirs(gen_paths['reinforced_data_dir'], exist_ok=True)
            
            if os.path.isfile(gen_paths['reinforced_data_file']):
                print(f"文件已存在，跳过: {gen_paths['reinforced_data_file']}")
                continue
                
            _generate_and_save_texts(data_loader, tep, gen_paths['reinforced_data_file'], args.max_batches)
    else:
        # 验证集和测试集：生成单个文本用于评测
        os.makedirs(paths['reinforced_data_dir'], exist_ok=True)
        
        if not os.path.isfile(paths['reinforced_data_file']):
            _generate_and_save_texts(data_loader, tep, paths['reinforced_data_file'], args.max_batches)
        else:
            print(f"文件已存在，跳过: {paths['reinforced_data_file']}")
    
    # 清理内存
    del df_all
    clear_dataloder(data_loader)
    torch.cuda.empty_cache()
    gc.collect()


def _generate_and_save_texts(data_loader, tep, save_path, max_batches=None):
    """生成并保存增强文本数据
    
    Args:
        data_loader: 数据加载器
        tep: 文本增强模型
        save_path: 保存路径
        max_batches: 最大处理batch数量，None表示处理所有batch
    """
    all_data = {
        'history_series': [], 'horizon_series': [], 
        'prompt': [], 'reinforced_text': []
    }
    
    batch_count = 0
    desc = f"生成增强文本{'(限制'+str(max_batches)+'批次)' if max_batches else ''}"
    
    for prompts, history_series, horizon_series in tqdm(data_loader, desc=desc):
        reinforced_texts = tep.get_model_response(prompts)
        
        all_data['history_series'].extend(history_series)
        all_data['horizon_series'].extend(horizon_series)
        all_data['prompt'].extend(prompts)
        all_data['reinforced_text'].extend(reinforced_texts)
        
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            print(f"已处理 {batch_count} 个batch，达到限制数量，停止生成")
            break
    
    pd.DataFrame(all_data).to_csv(save_path, index=False)
    print(f"已保存: {save_path} (共处理 {batch_count} 个batch)")


def multimodal_prediction(args, tsf_model, flag="train", text_type="reinforced_text", gen_idx=0):
    """
    多模态时间序列预测，生成奖励数据
    
    Args:
        args: 参数配置
        tsf_model: 时间序列预测模型
        flag: 数据集类型
        text_type: 文本类型
        gen_idx: 生成索引
    """
    # 获取路径
    paths = get_data_paths(args, flag, gen_idx)
    
    # 设置保存路径
    os.makedirs(paths['reward_data_dir'], exist_ok=True)
    
    if os.path.isfile(paths['reward_data_file']):
        print(f"奖励数据已存在，跳过: {paths['reward_data_file']}")
        return
    
    # 加载增强文本数据
    df_all = pd.read_csv(paths['reinforced_data_file'])
    df_all = df_all.dropna(axis=0, how='any', subset=[text_type])
    df_all.reset_index(drop=True, inplace=True)
    
    # 创建数据集
    pred_dataset = _create_prediction_dataset(args, df_all, text_type, tsf_model)
    pred_loader = DataLoader(pred_dataset, args.batch_size, shuffle=False, num_workers=128, drop_last=False)
    
    # 执行预测并计算奖励
    all_data_with_reward = _execute_prediction_and_reward(args, tsf_model, pred_loader, text_type)
    
    # 保存结果
    pd.DataFrame(all_data_with_reward).to_csv(paths['reward_data_file'], index=False)
    print(f"已保存奖励数据: {paths['reward_data_file']}")
    
    # 清理内存
    del df_all, all_data_with_reward, pred_dataset
    clear_dataloder(pred_loader)
    torch.cuda.empty_cache()
    gc.collect()


def _create_prediction_dataset(args, df_all, text_type, tsf_model):
    """创建预测数据集"""
    if args.tsf_type == "mcd-tsf":
        return MCD_dataset(df_all, [args.hist_len, args.pred_len], args.data_name, text_type, down_sample=args.down_sample)
    elif args.tsf_type == "tfhts":
        model_id = "/data2/user2/Llama-3.1-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        text_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
        )
        return TFHTSDataset(df_all, args.hist_len, args.pred_len, text_type, tokenizer, text_model, args.data_name, device=tsf_model.device)
    else:
        return Time_LLM_Dataset(df_all, args.hist_len, args.pred_len, args.data_name, text_type)


def _execute_prediction_and_reward(args, tsf_model, pred_loader, text_type):
    """执行预测并计算奖励"""
    all_data_with_reward = {
        'history_series': [], 'horizon_series': [], 
        'prompt': [], text_type: [], 'reward1': []
    }
    
    for batch_data in tqdm(pred_loader, desc="执行预测"):
        # 执行预测
        pred_series = _get_prediction(args, tsf_model, batch_data)
        
        # 处理批次维度
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
        
        # 计算奖励（MSE的负值）
        reward = -np.mean((pred_series.cpu().numpy() - horizon_series.cpu().numpy()) ** 2, axis=1)
        all_data_with_reward['reward1'].extend(reward.tolist())
    
    return all_data_with_reward


def _get_prediction(args, tsf_model, batch_data):
    """获取预测结果"""
    if args.tsf_type == "mcd-tsf":
        return tsf_model.mcd_tsf_predict_(batch_data, 3)
    elif args.tsf_type == "tfhts":
        return tsf_model.tfhts_predict_(batch_data["text_emb"], batch_data["history_series"]).squeeze()
    else:
        dec_inp = torch.zeros_like(batch_data["horizon_series"]).float().to(tsf_model.device)
        return tsf_model.time_llm_predict_(batch_data["history_series"], batch_data["seq_x_mark"], 
                                         dec_inp, batch_data["seq_y_mark"], batch_data["prompt"])


def get_preference_data(args, original_perf_df):
    """
    生成偏好数据集用于DPO训练
    
    流程：
    1. 加载所有生成的奖励数据
    2. 与原始性能对比，筛选有效文本
    3. 计算综合奖励（预测奖励 + 文本质量奖励）
    4. 选择最佳和最差文本作为偏好对
    5. 生成DPO训练格式的数据
    6. 统计抛弃率
    
    Args:
        args: 参数配置
        original_perf_df: 原始性能数据框
    """
    # 设置保存路径
    paths = get_data_paths(args)
    os.makedirs(paths['preference_data'], exist_ok=True)
    preference_path = os.path.join(
        paths['preference_data'], 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_genNum{args.gen_num}_iter{args.iter_idx}_{args.exp_time}.json"
    )
    
    if os.path.isfile(preference_path):
        print(f"偏好数据已存在，跳过: {preference_path}")
        return
    
    # 加载并合并所有奖励数据
    df1 = _load_reward_data(args, 0)
    for gen_idx in range(1, args.gen_num):
        df2 = _load_reward_data(args, gen_idx)
        df1 = _merge_reward_data(df1, df2, gen_idx)
    
    # 加载上一轮的性能数据（如果存在）
    prev_perf_df = get_previous_iteration_performance(args)
    
    # 合并上一轮性能数据
    if prev_perf_df is not None:
        # 确保df1有history_series_str列
        if 'history_series_str' not in df1.columns:
            df1['history_series_str'] = df1['history_series'].astype(str)
        # 检查上一轮数据是否有history_series_str列，如果没有则添加
        if 'history_series_str' not in prev_perf_df.columns:
            prev_perf_df['history_series_str'] = prev_perf_df['history_series'].astype(str)
        df1 = pd.merge(df1, prev_perf_df[['history_series_str', 'mse']], 
                       on='history_series_str', how='left')
        df1 = df1.rename(columns={'mse': 'prev_mse'})
    
    # 与原始性能对比，筛选有效文本
    df1 = _filter_effective_texts(df1, original_perf_df, args.gen_num, args)
    
    # 检查是否有有效数据
    if len(df1) == 0:
        print(f"警告：第{args.iter_idx}轮训练中所有生成文本都被抛弃，跳过该轮训练")
        sys.exit(1)  # 使用退出码1表示需要跳过该轮训练（没有有效样本）
    
    # 计算有效数据比例
    total_samples = len(_load_reward_data(args, 0))  # 获取原始数据量
    effective_ratio = len(df1) / total_samples
    print(f"有效数据比例: {effective_ratio:.2%} ({len(df1)}/{total_samples})")
    
    # 添加文本质量奖励（可选）
    if not args.disable_text_quality_reward:
        print("添加文本质量奖励...")
        for gen_idx in range(args.gen_num):
            total_quality_score = 0
            for row_idx in range(len(df1)):
                score = calculate_normalized_scores(df1.loc[row_idx, f"reinforced_text_{gen_idx}"])
                df1.loc[row_idx, f"reward_{gen_idx}"] += args.lam * score
                total_quality_score += score
            
            print(f"  生成{gen_idx}的平均文本质量分数: {total_quality_score/len(df1):.4f}")
    else:
        print("跳过文本质量奖励计算")
    
    # 验证奖励计算
    print("验证奖励计算...")
    reward_cols = [c for c in df1.columns if 'reward' in c]
    for col in reward_cols:
        print(f"  {col}: 范围[{df1[col].min():.4f}, {df1[col].max():.4f}], 均值{df1[col].mean():.4f}")
    
    # 生成偏好对
    preference_dataset = _generate_preference_pairs(df1, args.gen_num)
    
    # 保存偏好数据
    with open(preference_path, 'w') as f:
        json.dump(preference_dataset, f, indent=4)
    
    # 更新LLaMA-Factory数据集配置
    _update_dataset_info(args, preference_path)
    
    print(f"偏好数据已保存: {preference_path}")
    print(f"有效数据比例: {effective_ratio:.2%}")


def _filter_effective_texts(df1, original_perf_df, gen_num, args):
    """
    筛选有效的生成文本
    
    Args:
        df1: 包含所有生成文本奖励的数据框
        original_perf_df: 原始性能数据框
        gen_num: 生成数量
        args: 参数配置
    
    Returns:
        pd.DataFrame: 筛选后的有效数据框
    """
    print("开始筛选有效文本...")
    
    # 将history_series转换为字符串以便匹配（pandas无法直接合并列表格式）
    df1['history_series_str'] = df1['history_series'].astype(str)
    original_perf_df['history_series_str'] = original_perf_df['history_series'].astype(str)
    
    # 合并原始性能数据
    df1 = pd.merge(df1, original_perf_df[['history_series_str', 'original_mse']], 
                   on='history_series_str', how='left')
    
    # 加载上一轮的性能数据（如果存在）
    prev_perf_df = get_previous_iteration_performance(args)
    
    # 合并上一轮性能数据
    if prev_perf_df is not None:
        # 确保df1有history_series_str列
        if 'history_series_str' not in df1.columns:
            df1['history_series_str'] = df1['history_series'].astype(str)
        # 检查上一轮数据是否有history_series_str列，如果没有则添加
        if 'history_series_str' not in prev_perf_df.columns:
            prev_perf_df['history_series_str'] = prev_perf_df['history_series'].astype(str)
        df1 = pd.merge(df1, prev_perf_df[['history_series_str', 'mse']], 
                       on='history_series_str', how='left')
        df1 = df1.rename(columns={'mse': 'prev_mse'})
    
    # 筛选有效文本 - 综合考虑原始性能和上一轮性能
    effective_mask = pd.Series([False] * len(df1))
    performance_improvement_count = 0
    acceptable_degradation_count = 0
    iteration_improvement_count = 0
    
    for idx in range(len(df1)):
        # 检查该样本的所有生成文本是否都有效
        sample_effective = False
        best_improvement_vs_original = float('-inf')
        best_improvement_vs_prev = float('-inf')
        
        original_mse = df1.loc[idx, 'original_mse']
        prev_mse = df1.loc[idx, 'prev_mse'] if prev_perf_df is not None else None
        
        # 如果prev_mse是Series，取第一个值
        if hasattr(prev_mse, 'iloc'):
            prev_mse = prev_mse.iloc[0]

        for gen_idx in range(gen_num):
            # 计算生成文本的MSE（reward1是MSE的负值）
            generated_mse = -df1.loc[idx, f'reward_{gen_idx}']
            
            # 计算相对于原始文本的改进比例
            improvement_vs_original = (original_mse - generated_mse) / original_mse
            best_improvement_vs_original = max(best_improvement_vs_original, improvement_vs_original)
            
            # 计算相对于上一轮的改进比例
            if prev_mse is not None:
                improvement_vs_prev = (prev_mse - generated_mse) / prev_mse
                best_improvement_vs_prev = max(best_improvement_vs_prev, improvement_vs_prev)
            
            # 综合筛选条件（放宽条件）：
            # 1. 相对于原始文本：改进超过2% 或 下降不超过50%
            # 2. 相对于上一轮：改进超过1% 或 下降不超过30%
            # 3. 如果上一轮数据不存在，只考虑相对于原始文本的改进
            
            # condition_original = improvement_vs_original > 0.02 or improvement_vs_original > -0.50
            condition_original = improvement_vs_original > -float('inf')
            condition_prev = True  # 默认通过
            
            if prev_mse is not None:
                # condition_prev = improvement_vs_prev > 0.01 or improvement_vs_prev > -0.30
                condition_prev = improvement_vs_prev > -float('inf')
            
            if condition_original and condition_prev:
                sample_effective = True
                
                # 统计改进类型
                if improvement_vs_original > 0.02:
                    performance_improvement_count += 1
                elif improvement_vs_original > -0.50:
                    acceptable_degradation_count += 1
                
                if prev_mse is not None and improvement_vs_prev > 0.01:
                    iteration_improvement_count += 1
                
                break
        
        effective_mask[idx] = sample_effective
        
        # 如果所有生成文本都无效，记录最佳改进情况
        if not sample_effective:
            print(f"样本{idx}被抛弃:")
            print(f"  相对于原始文本最佳改进: {best_improvement_vs_original:.2%}")
            if prev_mse is not None:
                print(f"  相对于上一轮最佳改进: {best_improvement_vs_prev:.2%}")
    
    # 筛选有效样本
    df1_filtered = df1[effective_mask].copy()
    
    # 清理临时列
    columns_to_drop = ['history_series_str', 'original_mse']
    if prev_perf_df is not None:
        columns_to_drop.append('prev_mse')
    df1_filtered = df1_filtered.drop(columns_to_drop, axis=1)
    
    # 重置索引确保连续性
    df1_filtered = df1_filtered.reset_index(drop=True)
    
    print(f"筛选结果统计:")
    print(f"  总样本数: {len(df1)}")
    print(f"  有效样本数: {len(df1_filtered)}")
    print(f"  有效比例: {len(df1_filtered)/len(df1):.2%}")
    print(f"  相对于原始文本改进样本: {performance_improvement_count}")
    print(f"  相对于原始文本可接受下降样本: {acceptable_degradation_count}")
    if prev_perf_df is not None:
        print(f"  相对于上一轮改进样本: {iteration_improvement_count}")
    
    return df1_filtered


def _load_reward_data(args, gen_idx):
    """加载奖励数据"""
    paths = get_data_paths(args, "train", gen_idx)
    df = pd.read_csv(paths['reward_data_file'])
    df[f'reward_{gen_idx}'] = df.iloc[:, 4:].sum(axis=1)
    # 重置索引确保连续性
    df = df.reset_index(drop=True)
    # 确保列名正确，同时保留prompt列
    result_df = df[['history_series', 'prompt', 'reinforced_text', f'reward_{gen_idx}']].copy()
    result_df = result_df.rename(columns={'reinforced_text': f'reinforced_text_{gen_idx}'})
    return result_df


def _merge_reward_data(df1, df2, gen_idx):
    """合并奖励数据"""
    merged_df = pd.merge(df1, df2, how='inner', on='history_series')
    
    # 处理合并后的列名
    # 如果存在prompt_x和prompt_y，保留prompt_x（第一个数据框的prompt）
    if 'prompt_x' in merged_df.columns and 'prompt_y' in merged_df.columns:
        merged_df['prompt'] = merged_df['prompt_x']
        merged_df = merged_df.drop(['prompt_x', 'prompt_y'], axis=1)
    elif 'prompt_x' in merged_df.columns:
        merged_df['prompt'] = merged_df['prompt_x']
        merged_df = merged_df.drop('prompt_x', axis=1)
    elif 'prompt_y' in merged_df.columns:
        merged_df['prompt'] = merged_df['prompt_y']
        merged_df = merged_df.drop('prompt_y', axis=1)
    
    # 重置索引确保连续性
    return merged_df.reset_index(drop=True)


def _generate_preference_pairs(df1, gen_num):
    """生成偏好对"""
    reward_cols = [c for c in df1.columns if 'reward' in c]
    reward_data = df1[reward_cols]
    
    # 选择最佳和最差文本
    # reward是MSE的负值，所以reward最大对应MSE最小
    pos = reward_data.idxmax(axis=1).apply(lambda x: 'reinforced_text'+x[-2:])
    neg = reward_data.idxmin(axis=1).apply(lambda x: 'reinforced_text'+x[-2:])
    
    # 添加调试信息
    print(f"偏好对生成统计:")
    print(f"  总样本数: {len(df1)}")
    print(f"  奖励列: {reward_cols}")
    print(f"  平均奖励值: {reward_data.mean().to_dict()}")
    print(f"  奖励值范围: {reward_data.min().min():.4f} ~ {reward_data.max().max():.4f}")
    
    preference_dataset = []
    for idx in range(len(pos)):
        # 获取实际的奖励值用于验证
        pos_reward = reward_data.loc[idx, pos[idx].replace('reinforced_text_', 'reward_')]
        neg_reward = reward_data.loc[idx, neg[idx].replace('reinforced_text_', 'reward_')]
        
        # 验证选择逻辑
        if pos_reward <= neg_reward:
            print(f"警告: 样本{idx}的偏好对选择可能有问题 - pos_reward({pos_reward:.4f}) <= neg_reward({neg_reward:.4f})")
        
        preference_dataset.append({
            "conversations": [{"from": "human", "value": df1.loc[idx, 'prompt']}],
            "chosen": {"from": "gpt", "value": df1.loc[idx, pos[idx]]},
            "rejected": {"from": "gpt", "value": df1.loc[idx, neg[idx]]}
        })
    
    return preference_dataset


def _update_dataset_info(args, preference_path):
    """更新LLaMA-Factory数据集配置"""
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
    """主函数：执行完整的数据准备流程"""
    parser = argparse.ArgumentParser(description="TeR-TSF 数据准备阶段")
    parser.add_argument("--data_dir", type=str, default="/data2/user2/ter_tsf")
    parser.add_argument("--data_name", type=str, default="Agriculture")
    parser.add_argument("--llm_type", type=str, default='qwen3-1.7b')
    parser.add_argument("--tsf_type", type=str, default='mcd_tsf')
    parser.add_argument("--hist_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gen_num", type=int, default=2, help="每个轮次生成的文本数量，可以是任意正整数")
    parser.add_argument("--iter_idx", type=int, default=0)
    parser.add_argument("--llm_path", type=str, default="original")
    parser.add_argument("--exp_time", type=str, default="")
    parser.add_argument("--down_sample", type=int, default=0)
    parser.add_argument("--llama_factory_dir", type=str, default="/home/user2/projects/TeR_TSF/llama-factory-main")
    parser.add_argument("--lam", type=float, default=1.5)
    parser.add_argument("--disable_text_quality_reward", action="store_true", help="禁用文本质量奖励（默认启用）")
    parser.add_argument("--max_batches", type=int, default=None, help="限制处理的最大batch数量（用于快速测试，None表示处理所有batch）")
    args = parser.parse_args()
    
    print("=== 开始数据准备阶段 ===")
    print(f"LLM类型: {args.llm_type}")
    print(f"TSF类型: {args.tsf_type}")
    print(f"数据集: {args.data_name}")
    print(f"历史长度: {args.hist_len}")
    print(f"预测长度: {args.pred_len}")
    print(f"迭代轮次: {args.iter_idx}")
    print(f"数据目录: {args.data_dir}")
    print(f"LLM路径: {args.llm_path}")
    if args.max_batches is not None:
        print(f"⚠️  批次限制: 仅处理前 {args.max_batches} 个batch（快速测试模式）")
    print("================================")
    
    # 检查原始数据文件是否存在
    original_data_path = os.path.join(
        "/data2/user2/ter_tsf/processed_data", 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_train.csv"
    )
    print(f"检查原始数据文件: {original_data_path}")
    if not os.path.exists(original_data_path):
        print(f"错误：原始数据文件不存在: {original_data_path}")
        sys.exit(2)  # 使用退出码2表示文件不存在错误
    
    # 初始化模型
    print("1. 初始化文本增强模型...")
    try:
        tep = TextReinforcementModel(llm_name=args.llm_type, llm_path=args.llm_path)
        print("文本增强模型初始化成功")
    except Exception as e:
        print(f"文本增强模型初始化失败: {e}")
        sys.exit(3)  # 使用退出码3表示模型初始化错误
    
    print("2. 初始化时间序列预测模型...")
    try:
        tsf_model = _init_tsf_model(args)
        print("时间序列预测模型初始化成功")
    except Exception as e:
        print(f"时间序列预测模型初始化失败: {e}")
        sys.exit(4)  # 使用退出码4表示TSF模型初始化错误
    
    # 评估原始性能
    print("3. 评估原始数据性能...")
    try:
        original_perf_df = evaluate_original_performance(args, tsf_model)
        print("原始性能评估完成")
    except Exception as e:
        print(f"原始性能评估失败: {e}")
        sys.exit(5)  # 使用退出码5表示性能评估错误
    
    # 生成增强文本
    print("4. 生成增强文本数据...")
    gen_reinforced_text(args, tep, flag="train")
    # 生成验证集和测试集的文本
    # 第0轮：生成未训练基线
    # 第1轮及以后：生成当前轮训练后的文本
    if args.iter_idx == 0:
        # print("   - 生成验证集未训练基线...")
        gen_reinforced_text(args, tep, flag="val", trained=False)
        print("   - 生成测试集未训练基线...")
        gen_reinforced_text(args, tep, flag="test", trained=False)
    else:
        # print("   - 生成验证集当前轮文本...")
        gen_reinforced_text(args, tep, flag="val", trained=True)
        print("   - 生成测试集当前轮文本...")
        gen_reinforced_text(args, tep, flag="test", trained=True)
    
    # 清理文本增强模型
    del tep
    torch.cuda.empty_cache()
    gc.collect()
    
    # 多模态预测生成奖励数据
    print("5. 执行多模态预测生成奖励数据...")
    for gen_idx in range(args.gen_num):
        multimodal_prediction(args, tsf_model, flag="train", text_type="reinforced_text", gen_idx=gen_idx)
    
    # 生成偏好数据
    print("6. 生成偏好数据集...")
    get_preference_data(args, original_perf_df)
    
    # 记录当前轮次的性能数据（用于下一轮比较）
    print("7. 记录性能数据...")
    gen0_reward_data = _load_reward_data(args, 0)
    record_iteration_performance(args, gen0_reward_data, gen_idx=0)
    
    # 分析训练进展
    print("8. 分析训练进展...")
    analyze_training_progress(args)
    
    print("=== 数据准备阶段完成 ===")


def _init_tsf_model(args):
    """初始化时间序列预测模型"""
    if args.tsf_type == "mcd-tsf":
        return BaseMultimodalTSFModel(args.hist_len, args.pred_len, args.data_name, 
                                    bm_name=args.tsf_type, freq=stat_dict[args.data_name]['freq'])
    elif args.tsf_type == "tfhts":
        return BaseMultimodalTSFModel(args.hist_len, args.pred_len, args.data_name, bm_name="textfusionhts", llm_type=args.llm_type, iter_=args.iter_idx, exp_time=args.exp_time)
    elif args.tsf_type == "time-llm":
        return BaseMultimodalTSFModel(args.hist_len, args.pred_len, args.data_name, bm_name="time-llm")
    else:
        raise ValueError(f"未定义的时间序列预测模型: {args.tsf_type}")


def record_iteration_performance(args, df1, gen_idx=1):
    """
    记录每轮训练后的性能数据，用于后续轮次的比较
    
    Args:
        args: 参数配置
        df1: 包含奖励数据的数据框
        gen_idx: 生成索引（通常使用gen1，即训练后的生成）
    """
    # 创建性能记录目录，添加hist_len、pred_len和exp_time信息
    perf_record_dir = os.path.join(
        args.data_dir, args.llm_type, args.tsf_type, args.data_name,
        "iteration_performance", f"{args.hist_len}_{args.pred_len}_{args.exp_time}"
    )
    os.makedirs(perf_record_dir, exist_ok=True)
    
    # 记录当前轮次的性能
    perf_record_path = os.path.join(
        perf_record_dir,
        f"iter{args.iter_idx}_performance.csv"
    )
    
    # 提取性能数据 - 使用正确的列名
    reward_col = f'reward_{gen_idx}'
    
    perf_data = df1[['history_series', reward_col]].copy()
    perf_data['mse'] = -perf_data[reward_col]  # 转换为MSE
    perf_data['iter_idx'] = args.iter_idx
    perf_data['gen_idx'] = gen_idx
    
    # 添加history_series_str列用于后续匹配
    perf_data['history_series_str'] = perf_data['history_series'].astype(str)
    
    # 保存性能记录
    perf_data.to_csv(perf_record_path, index=False)
    print(f"已记录第{args.iter_idx}轮性能数据: {perf_record_path}")
    
    # 计算并显示性能统计
    mean_mse = perf_data['mse'].mean()
    std_mse = perf_data['mse'].std()
    print(f"第{args.iter_idx}轮平均MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    
    return perf_data


def get_previous_iteration_performance(args):
    """
    获取上一轮的性能数据
    
    Args:
        args: 参数配置
    
    Returns:
        pd.DataFrame: 上一轮性能数据，如果不存在则返回None
    """
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
            # 检查是否有history_series_str列，如果没有则添加
            if 'history_series_str' not in prev_perf_df.columns:
                prev_perf_df['history_series_str'] = prev_perf_df['history_series'].astype(str)
            print(f"成功加载第{args.iter_idx-1}轮性能数据: {len(prev_perf_df)} 样本")
            return prev_perf_df
        except Exception as e:
            print(f"加载第{args.iter_idx-1}轮性能数据失败: {e}")
            return None
    else:
        print(f"第{args.iter_idx-1}轮性能数据文件不存在: {prev_perf_path}")
        return None


def analyze_training_progress(args):
    """
    分析训练进展，比较不同轮次的性能
    
    Args:
        args: 参数配置
    """
    perf_record_dir = os.path.join(
        args.data_dir, args.llm_type, args.tsf_type, args.data_name,
        "iteration_performance", f"{args.hist_len}_{args.pred_len}_{args.exp_time}"
    )
    
    if not os.path.exists(perf_record_dir):
        print("性能记录目录不存在，无法分析训练进展")
        return
    
    # 加载所有轮次的性能数据
    all_perf_data = []
    for iter_idx in range(args.iter_idx + 1):
        perf_path = os.path.join(perf_record_dir, f"iter{iter_idx}_performance.csv")
        if os.path.exists(perf_path):
            perf_df = pd.read_csv(perf_path)
            perf_df['iter_idx'] = iter_idx
            all_perf_data.append(perf_df)
    
    if not all_perf_data:
        print("没有找到性能记录数据")
        return
    
    # 合并所有性能数据
    combined_perf = pd.concat(all_perf_data, ignore_index=True)
    
    # 计算每轮的平均MSE
    iter_summary = combined_perf.groupby('iter_idx')['mse'].agg(['mean', 'std', 'count']).reset_index()
    
    print("\n=== 训练进展分析 ===")
    print("轮次\t平均MSE\t\t标准差\t\t样本数")
    print("-" * 50)
    
    for _, row in iter_summary.iterrows():
        print(f"iter{row['iter_idx']}\t{row['mean']:.6f}\t{row['std']:.6f}\t{row['count']}")
    
    # 计算改进情况
    if len(iter_summary) > 1:
        print("\n=== 轮次间改进情况 ===")
        for i in range(1, len(iter_summary)):
            prev_mse = iter_summary.iloc[i-1]['mean']
            curr_mse = iter_summary.iloc[i]['mean']
            improvement = (prev_mse - curr_mse) / prev_mse * 100
            
            if improvement > 0:
                print(f"iter{i-1} → iter{i}: 改进 {improvement:.2f}%")
            else:
                print(f"iter{i-1} → iter{i}: 下降 {abs(improvement):.2f}%")
    
    # 与原始性能比较
    original_perf_path = os.path.join(
        args.data_dir, args.llm_type, args.tsf_type, args.data_name,
        "original_performance", 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.tsf_type}_train_mse.csv"
    )
    
    if os.path.exists(original_perf_path):
        original_perf = pd.read_csv(original_perf_path)
        original_mean_mse = original_perf['original_mse'].mean()
        
        print(f"\n=== 与原始性能比较 ===")
        print(f"原始平均MSE: {original_mean_mse:.6f}")
        
        for _, row in iter_summary.iterrows():
            improvement = (original_mean_mse - row['mean']) / original_mean_mse * 100
            if improvement > 0:
                print(f"iter{row['iter_idx']}: 相比原始改进 {improvement:.2f}%")
            else:
                print(f"iter{row['iter_idx']}: 相比原始下降 {abs(improvement):.2f}%")
    
    print("=" * 50)


if __name__ == "__main__":
    main()