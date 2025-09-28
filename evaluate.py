"""
TeR-TSF 评估阶段脚本

逻辑流程图：
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   增强文本数据   │    │   多模态预测      │    │   性能评估       │
│ (reinforced_data)│───▶│ (时间序列预测)    │───▶│  (MSE/MAE)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   结果记录        │    │   实验对比       │
                       │(results.json)    │    │(不同文本类型)    │
                       └──────────────────┘    └─────────────────┘

文件路径结构：
/data_dir/
├── {llm_type}/                    # 按LLM类型分类
│   ├── {tsf_type}/               # 按TSF类型分类
│   │   ├── {data_name}/          # 按数据集分类
│   │   │   ├── reinforced_data/  # 增强文本数据
│   │   │   │   ├── val/          # 验证集
│   │   │   │   │   ├── untrained.csv  # 未训练模型生成（固定基准线）
│   │   │   │   │   └── iter{i}.csv    # 第i轮训练后生成
│   │   │   │   └── test/         # 测试集
│   │   │   │       ├── untrained.csv  # 未训练模型生成（固定基准线）
│   │   │   │       └── iter{i}.csv    # 第i轮训练后生成
│   │   │   └── results/          # 评估结果
│   │   │       ├── val.json      # 验证集结果
│   │   │       └── test.json     # 测试集结果
│   │   └── ...
│   └── ...
└── ...

评估逻辑说明：
1. 固定基准线评估：
   - original_text: 原始文本（固定基准线）
   - reinforced_text (untrained): 未训练模型生成的文本（固定基准线）

2. 训练效果评估：
   - reinforced_text (trained): 当前轮训练后模型生成的文本
   - 与固定基准线对比，评估训练效果

3. 评估流程：
   - 第0轮：只评估固定基准线（未训练模型 = 当前轮模型）
   - 第1轮及以后：评估固定基准线 + 当前轮训练效果

主要功能：
1. 使用训练后的LLM生成增强文本
2. 结合时间序列数据进行多模态预测
3. 计算预测性能指标（MSE、MAE）
4. 对比不同文本类型的预测效果
5. 记录实验结果用于分析

输入：增强文本数据 + 时间序列数据
输出：预测性能指标 + 实验结果记录
"""

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

# 设置随机种子
SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_untrained_baseline_path(args, flag):
    """
    获取未训练基线的路径
    
    Args:
        args: 参数配置
        flag: 数据集类型 ("val"/"test")
    
    Returns:
        str: 未训练基线文件路径
    
    说明：
    - 未训练基线是固定的基准线，用于评估训练效果
    - 路径：untrained/{data}_{hist}_{pred}_{exp}.csv
    - 这个文件在训练开始前生成，作为所有轮次训练的对比基准
    """
    # 基础路径结构：{data_dir}/{llm_type}/{tsf_type}/{data_name}/
    base_path = os.path.join(args.data_dir, args.llm_type, args.tsf_type, args.data_name)
    
    # 使用untrained目录作为未训练基线
    reinforced_data_dir = os.path.join(base_path, "reinforced_data", flag, "untrained")
    baseline_path = os.path.join(
        reinforced_data_dir, 
        f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv"
    )
    
    return baseline_path


def evaluate_prediction(args, tsf_model, flag="test", text_type="reinforced_text", trained=False):
    """
    评估预测性能
    
    Args:
        args: 参数配置
        tsf_model: 时间序列预测模型
        flag: 数据集类型 ("val"/"test")
        text_type: 文本类型 ("reinforced_text"/"original_text")
        trained: 是否使用训练后的模型
    
    评估逻辑：
    1. original_text: 使用原始文本（固定基准线）
    2. reinforced_text + trained=False: 使用未训练模型生成的文本（固定基准线）
    3. reinforced_text + trained=True: 使用当前轮训练后模型生成的文本（训练效果）
    
    路径选择：
    - original_text: 直接使用原始数据路径
    - reinforced_text + trained=False: 使用untrained目录
    - reinforced_text + trained=True: 使用iter{iter_idx}目录
    """
    # 获取路径
    if text_type == "original_text":
        # 原始文本：直接使用原始数据路径
        original_data_path = os.path.join(
            "/data2/user2/ter_tsf/processed_data", 
            f"{args.data_name}_{args.hist_len}_{args.pred_len}_{flag}.csv"
        )
        paths = get_data_paths(args, flag, trained=trained)
        paths['reinforced_data_file'] = original_data_path
    elif not trained and text_type == "reinforced_text":
        # 未训练模型生成的文本：使用第0轮第0次生成的文件
        reinforced_data_file = get_untrained_baseline_path(args, flag)
        paths = get_data_paths(args, flag, trained=trained)
        paths['reinforced_data_file'] = reinforced_data_file
    else:
        # 当前轮训练模型生成的文本：使用当前轮生成的文件
        paths = get_data_paths(args, flag, trained=trained)
    
    # 设置结果保存路径
    result_save_path = os.path.join(paths['base'], "results")
    os.makedirs(result_save_path, exist_ok=True)
    
    # 确定迭代索引和文本类型标识
    if text_type == "original_text":
        iter_idx = -1  # 原始文本用-1表示
        record_text_type = "original_text"
    elif not trained:
        iter_idx = 0  # 未训练模型用0表示
        record_text_type = "reinforced_text (untrained)"
    else:
        iter_idx = args.iter_idx
        record_text_type = "reinforced_text (trained)"
    
    # 设置实验参数
    fixed_params = {
        'Data': args.data_name, 'hist_len': args.hist_len, 'pred_len': args.pred_len
    }
    varying_params = {
        'iter': iter_idx, 'llm': args.llm_type, 'tsf': args.tsf_type, 
        'gen_num': args.gen_num, 'dpo_lr': args.dpo_lr, 'dpo_epoch': args.dpo_epoch, 
        'text': record_text_type
    }
    
    # 检查是否已存在结果
    record_result = RecordExpMetrics(os.path.join(result_save_path, f"{flag}.json"))
    result_label = f"hist{args.hist_len}_pred{args.pred_len}_{args.llm_type}_{args.tsf_type}_gen{args.gen_num}_iter{iter_idx}_{text_type}_{trained}"
    
    if record_result.is_exist(fixed_params, varying_params):
        print(f"实验结果已存在，跳过: {result_label}-Experiment{args.exp_time}")
        return
    
    # 加载数据集
    if text_type == "original_text":
        # 使用原始数据
        df_all = pd.read_csv(paths['reinforced_data_file'])
        # 原始数据中已包含original_text列，无需添加占位符
    else:
        # 使用增强文本数据
        df_all = pd.read_csv(paths['reinforced_data_file'])
        df_all[[text_type]] = df_all[[text_type]].fillna("No text.")
    
    # 创建预测数据集
    pred_dataset = _create_evaluation_dataset(args, df_all, text_type, tsf_model)
    pred_loader = DataLoader(pred_dataset, args.batch_size, shuffle=False, num_workers=128, drop_last=False)
    
    # 执行预测并计算指标
    all_mse, all_mae = _execute_evaluation(args, tsf_model, pred_loader)
    
    # 计算平均指标
    mean_mse = np.mean(np.array(all_mse))
    mean_mae = np.mean(np.array(all_mae))
    
    # 记录结果
    result = {"MSE": mean_mse, "MAE": mean_mae}
    record_result.add_result(fixed_params, varying_params, result)
    
    print(f"评估完成 - MSE: {mean_mse:.6f}, MAE: {mean_mae:.6f}")
    
    # 清理内存
    del df_all, pred_dataset
    clear_dataloder(pred_loader)
    torch.cuda.empty_cache()
    gc.collect()


def _create_evaluation_dataset(args, df_all, text_type, tsf_model):
    """创建评估数据集"""
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


def _execute_evaluation(args, tsf_model, pred_loader):
    """执行评估计算"""
    all_mse, all_mae = [], []
    
    for batch_data in tqdm(pred_loader, desc="执行评估"):
        # 执行预测
        pred_series = _get_prediction(args, tsf_model, batch_data)
        
        # 处理批次维度 - 仿照prepare_stage.py的逻辑
        if batch_data['history_series'].size(0) == 1:
            # batch_size = 1的情况：保持二维结构
            horizon_series = batch_data['horizon_series'].squeeze().unsqueeze(dim=0)
            pred_series = pred_series.squeeze().unsqueeze(dim=0)
        else:
            # batch_size > 1的情况：直接squeeze
            horizon_series = batch_data['horizon_series'].squeeze()
            pred_series = pred_series.squeeze()
        
        # 计算指标 - 确保数组为二维，axis=1有效
        mse = np.mean((pred_series.cpu().numpy() - horizon_series.cpu().numpy()) ** 2, axis=1)
        mae = np.mean(np.abs(pred_series.cpu().numpy() - horizon_series.cpu().numpy()), axis=1)
        
        all_mse.extend(mse.tolist())
        all_mae.extend(mae.tolist())
    
    return all_mse, all_mae


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


def main():
    """主函数：执行完整的评估流程"""
    parser = argparse.ArgumentParser(description="TeR-TSF 评估阶段")
    parser.add_argument("--data_dir", type=str, default="/data2/user2/ter_tsf")
    parser.add_argument("--data_name", type=str, default="Agriculture")
    parser.add_argument("--llm_type", type=str, default='qwen3-1.7b')
    parser.add_argument("--tsf_type", type=str, default='mcd_tsf')
    parser.add_argument("--hist_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iter_idx", type=int, default=0)
    parser.add_argument("--llm_path", type=str, default="original")
    parser.add_argument("--exp_time", type=str, default="")
    parser.add_argument("--gen_num", type=int, default=2, help="每个轮次生成的文本数量，可以是任意正整数")
    parser.add_argument("--down_sample", type=int, default=1)
    parser.add_argument("--dpo_epoch", type=int, default=5)
    parser.add_argument("--dpo_lr", type=float, default=0.0001)
    args = parser.parse_args()

    print("=== 开始评估阶段 ===")
    print(f"LLM类型: {args.llm_type}")
    print(f"TSF类型: {args.tsf_type}")
    print(f"数据集: {args.data_name}")
    print(f"迭代轮次: {args.iter_idx}")
    print("================================")
    
    # 初始化模型
    print("1. 初始化文本增强模型...")
    tep = TextReinforcementModel(llm_name=args.llm_type, llm_path=args.llm_path)
    
    print("2. 初始化时间序列预测模型...")
    tsf_model = _init_tsf_model(args)
    
    # 执行评估
    print("3. 执行性能评估...")
    
    # 评估测试集
    print("   - 评估测试集...")
    
    # 1. 评估原始文本（固定基准线）
    print("     * 评估原始文本（基准线）...")
    evaluate_prediction(args, tsf_model, flag="test", text_type="original_text")
    
    # 2. 评估未训练模型生成的文本（固定基准线）
    # 注意：只有在第0轮时才评估未训练基线，后续轮次不需要重复评估
    if args.iter_idx == 0:
        print("     * 评估未训练模型生成的文本（基准线）...")
        evaluate_prediction(args, tsf_model, flag="test", text_type="reinforced_text", trained=False)
    else:
        print("     * 跳过未训练基线评估（已在第0轮完成）...")
    
    # 3. 评估当前轮训练模型生成的文本（训练效果）
    # 第0轮：生成第0轮训练后的测试集文本
    # 第1轮及以后：生成当前轮训练后的测试集文本
    print("     * 生成当前轮训练模型的增强文本...")
    gen_reinforced_text(args, tep, flag="test", trained=True)
    
    print("     * 评估当前轮训练模型生成的文本...")
    evaluate_prediction(args, tsf_model, flag="test", text_type="reinforced_text", trained=True)
    
    # 清理文本增强模型
    del tep
    torch.cuda.empty_cache()
    gc.collect()
    
    print("=== 评估阶段完成 ===")


if __name__ == "__main__":
    main()