#!/bin/bash

"""
TeR-TSF 主控制脚本

逻辑流程图：
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   参数解析       │    │   迭代训练循环    │    │   模型路径管理   │
│ (命令行参数)     │───▶│ (iter_num轮)     │───▶│ (原始→训练后)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   数据准备阶段    │    │ TextFusionHTS   │
                       │(prepare_stage.py)│───▶│   训练阶段      │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   DPO训练阶段    │    │   模型导出阶段   │
                       │(llamafactory-cli)│───▶│(export model)   │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   性能评估阶段   │    │   下一轮迭代    │
                       │(evaluate.py)     │───▶│(iter_idx++)     │
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
│   │   │   │   │   ├── untrained.csv  # 未训练模型生成（固定基准线）
│   │   │   │   │   └── iter{i}.csv    # 第i轮训练后生成
│   │   │   │   └── test/         # 测试集
│   │   │   │       ├── untrained.csv  # 未训练模型生成（固定基准线）
│   │   │   │       └── iter{i}.csv    # 第i轮训练后生成
│   │   │   ├── reward_data/      # 奖励数据
│   │   │   ├── preference_data/  # 偏好数据
│   │   │   ├── original_performance/  # 原始性能数据
│   │   │   └── results/          # 评估结果
│   │   └── ...
│   └── ...
└── ...

模型保存路径结构：
LLM模型路径：
/models/
├── {llm_type}/                    # 按LLM类型分类
│   ├── {tsf_type}/               # 按TSF类型分类
│   │   ├── {data_name}/          # 按数据集分类
│   │   │   ├── {hist_len}_{pred_len}_{exp_time}/  # 实验配置标识
│   │   │   │   ├── lora/         # LoRA适配器
│   │   │   │   │   └── iter{i}/  # 第i轮训练结果
│   │   │   │   └── merge/        # 合并后的模型
│   │   │   │       └── iter{i}/  # 第i轮合并结果
│   │   │   └── ...               # 其他实验配置
│   │   └── ...
│   └── ...
└── ...

TextFusionHTS模型路径：
/data2/user2/ter_tsf/
├── {llm_type}/                    # 按LLM类型分类
│   ├── tfhts/                    # TextFusionHTS专用目录
│   │   ├── {data_name}/          # 按数据集分类
│   │   │   ├── saved_models/     # 模型保存目录
│   │   │   │   ├── {data_name}_{hist_len}_{pred_len}_iter0_{timestamp}/  # 第0轮模型
│   │   │   │   ├── {data_name}_{hist_len}_{pred_len}_iter1_{timestamp}/  # 第1轮模型
│   │   │   │   └── ...           # 其他轮次模型
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...

训练逻辑说明：
1. 数据准备阶段：
   - 训练集：每个轮次生成gen_num个文本（gen_num可以是任意正整数），用于DPO偏好对比较
   - 验证集/测试集：生成固定基准线（未训练模型）和当前轮效果

2. TextFusionHTS训练阶段：
   - 第0轮：使用原始文本(original_text)训练TextFusionHTS模型
   - 第i轮(i>0)：使用增强文本(reinforced_text)训练TextFusionHTS模型
   - 训练数据：使用当前轮次生成的gen0数据作为训练集
   - 验证/测试数据：使用untrained基线数据进行评估

3. DPO训练阶段：
   - 使用当前轮生成的偏好数据进行训练
   - 偏好对：同一轮次内不同生成次数的文本质量比较

4. 模型路径管理：
   - 第0轮：使用原始模型
   - 第i轮：使用第(i-1)轮训练后的模型作为起点

5. 评估阶段：
   - 固定基准线：原始文本 + 未训练模型文本
   - 训练效果：当前轮训练后模型文本

主要功能：
1. 解析命令行参数并设置实验配置
2. 执行多轮迭代训练（DPO强化学习 + TextFusionHTS训练）
3. 每轮迭代包含：数据准备→TextFusionHTS训练→DPO训练→模型导出→性能评估
4. 管理模型路径（原始模型→训练后模型）
5. 自动处理实验时间和文件命名
6. 支持不同轮次使用不同文本类型（original_text vs reinforced_text）

输入：命令行参数（数据集、模型类型、训练参数等）
输出：训练后的模型 + TextFusionHTS模型 + 性能评估结果
"""

# 设置错误处理
set -e

# 参数解析函数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --llm_type) llm_type="$2"; shift ;;
            --tsf_type) tsf_type="$2"; shift ;;
            --data_name) data_name="$2"; shift ;;
            --hist_len) hist_len="$2"; shift ;;
            --pred_len) pred_len="$2"; shift ;;
            --exp_time) exp_time="$2"; shift ;;
            --batch_size) batch_size="$2"; shift ;;
            --gen_num) gen_num="$2"; shift ;;
            --iter_num) iter_num="$2"; shift ;;
            --lora_rank) lora_rank="$2"; shift ;;
            --lr) lr="$2"; shift ;;
            --num_train_epochs) num_train_epochs="$2"; shift ;;
            --per_device_train_batch_size) per_device_train_batch_size="$2"; shift ;;
            --llama_factory_dir) llama_factory_dir="$2"; shift ;;
            --down_sample) down_sample="$2"; shift ;;
            --disable_text_quality_reward) disable_text_quality_reward="--disable_text_quality_reward" ;;
            *) echo "未知参数: $1"; exit 1 ;;
        esac
        shift
    done
}

# 获取LLM配置
get_llm_config() {
    local llm_type=$1
    
    case "$llm_type" in
        "qwen3-1.7b")
            echo "qwen3:/data2/user2/Qwen3-1.7B"
            ;;
        "qwen3-8b")
            echo "qwen3:/data2/user2/Qwen3-8B"
            ;;
        "llama3.1-8b")
            echo "llama3:/data2/user2/Llama-3.1-8B"
            ;;
        "llama3.2-1b")
            echo "llama3:/data2/user2/Llama-3.2-1B"
            ;;
        "llama3.2-3b")
            echo "llama3:/data2/user2/Llama-3.2-3B"
            ;;
        "qwen3-4b")
            echo "qwen3:/data2/user2/Qwen3-4B"
            ;;
        *)
            echo "错误：无法识别的 llm_type 类型 '$llm_type'" >&2
            exit 1
            ;;
    esac
}

# 设置模型路径
setup_model_paths() {
    local iter_idx=$1
    local llm_template=$2
    local initial_path=$3
    
    # 构建包含hist_len、pred_len和exp_time的路径标识符
    local path_identifier="${hist_len}_${pred_len}_${exp_time}"
    
    # 模型路径管理逻辑：
    # - 第0轮：使用原始模型
    # - 第i轮：使用第(i-1)轮训练后的模型作为起点
    # - 如果找不到历史模型，回退到原始模型
    if (( iter_idx == 0 )); then
        llm_path="$initial_path"
    else
        local found_model=0
        local search_iter=$((iter_idx - 1))
        while (( search_iter >= 0 )); do
            local prev_merge_path="/data2/user2/ter_tsf/models/${llm_type}/${tsf_type}/${data_name}/${path_identifier}/merge/iter${search_iter}"
            if [ -d "$prev_merge_path" ]; then
                llm_path="$prev_merge_path"
                found_model=1
                echo "找到最近的历史模型: $llm_path"
                break
            fi
            search_iter=$((search_iter - 1))
        done
        if (( found_model == 0 )); then
            echo "警告：所有历史模型都不存在，使用初始模型"
            llm_path="$initial_path"
        fi
    fi
    
    local dataset_name="${data_name}_h${hist_len}_p${pred_len}_${llm_type}_${tsf_type}_genNum${gen_num}_iter${iter_idx}_${exp_time}"
    adapter_output_dir="/data2/user2/ter_tsf/models/${llm_type}/${tsf_type}/${data_name}/${path_identifier}/lora/iter${iter_idx}"
    merge_output_dir="/data2/user2/ter_tsf/models/${llm_type}/${tsf_type}/${data_name}/${path_identifier}/merge/iter${iter_idx}"
    
    echo "llm_path = ${llm_path}"
    echo "adapter_output_dir = ${adapter_output_dir}"
    echo "merge_output_dir = ${merge_output_dir}"
}

# 数据准备阶段
run_data_preparation() {
    local iter_idx=$1
    local llm_path=$2
    
    echo "===== 第 ${iter_idx} 轮生成偏好数据 ====="
    
    # 数据准备阶段功能：
    # 1. 使用当前模型生成gen_num个增强文本（gen_num可以是任意正整数）
    # 2. 计算多模态预测奖励
    # 3. 生成DPO训练用的偏好对
    # 4. 筛选有效样本（性能改进的文本）
    
    # 运行数据准备脚本并实时显示输出
    python prepare_stage.py \
        --data_dir /data2/user2/ter_tsf \
        --data_name $data_name \
        --llm_type $llm_type \
        --tsf_type $tsf_type \
        --hist_len $hist_len \
        --pred_len $pred_len \
        --batch_size $batch_size \
        --gen_num $gen_num \
        --iter_idx $iter_idx \
        --llm_path $llm_path \
        --exp_time $exp_time \
        --llama_factory_dir $llama_factory_dir \
        --down_sample $down_sample \
        $disable_text_quality_reward
    
    local exit_code=$?
    
    if (( exit_code == 0 )); then
        echo "数据准备阶段成功完成"
        return 0
    elif (( exit_code == 1 )); then
        echo "数据准备阶段：没有有效样本，跳过第 ${iter_idx} 轮训练"
        return 1  # 返回1表示需要跳过该轮
    else
        echo "数据准备阶段发生错误，退出码: ${exit_code}"
        echo "中断整个训练流程"
        exit $exit_code  # 直接退出整个程序
    fi
}

# TextFusionHTS训练阶段
run_tfhts_training() {
    local iter_idx=$1
    
    echo "===== 第 ${iter_idx} 轮TextFusionHTS训练 ====="
    
    # 确定text_type：第0轮使用original_text，其他轮次使用reinforced_text
    local text_type="original_text"
    if (( iter_idx > 0 )); then
        text_type="reinforced_text"
    fi
    
    # 设置数据目录和保存目录
    # 注意：reinforced_data目录结构与prepare_stage.py保持一致
    local data_dir="/data2/user2/ter_tsf/${llm_type}/${tsf_type}/${data_name}/reinforced_data"
    local save_dir="/data2/user2/ter_tsf/${llm_type}/tfhts/${data_name}/saved_models"
    
    # 根据数据集类型设置patch_len和stride参数
    local patch_len=16
    local stride=8

    # 如果是weather或Heart_Rate数据集，使用更小的patch_size和stride
    if [[ "$data_name" == "weather"* ]] || [[ "$data_name" == "Heart_Rate"* ]]; then
        patch_len=4
        stride=2
    fi
    
    echo "训练参数:"
    echo "  - 数据目录: $data_dir"
    echo "  - 保存目录: $save_dir"
    echo "  - 文本类型: $text_type"
    echo "  - 数据集: $data_name"
    echo "  - 历史长度: $hist_len"
    echo "  - 预测长度: $pred_len"
    echo "  - Patch长度: $patch_len"
    echo "  - Stride: $stride"
    
    # 运行TextFusionHTS训练脚本
    python ./Models/TextFusionHTS/train_tfhts.py \
        --data_dir $data_dir \
        --save_dir $save_dir \
        --data_name $data_name \
        --hist_len $hist_len \
        --pred_len $pred_len \
        --batch_size 32 \
        --text_type $text_type \
        --patch_len $patch_len \
        --stride $stride \
        --epochs 200 \
        --lr 1e-3 \
        --exp_time $exp_time \
        --iter_idx $iter_idx
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "TextFusionHTS训练成功完成"
    else
        echo "TextFusionHTS训练失败，退出码: $exit_code"
        exit 1
    fi
}

# DPO训练阶段
run_dpo_training() {
    local iter_idx=$1
    local llm_path=$2
    local llm_template=$3
    local dataset_name=$4
    
    echo "===== 第 ${iter_idx} 轮训练 ====="
    echo "训练参数:"
    echo "  - 模型路径: $llm_path"
    echo "  - 数据集: $dataset_name"
    echo "  - 学习率: $lr"
    echo "  - 训练轮数: $num_train_epochs"
    echo "  - 批次大小: $per_device_train_batch_size"
    echo "  - LoRA rank: $lora_rank"
    
    # DPO训练阶段功能：
    # 1. 使用偏好对数据进行DPO训练
    # 2. 偏好对：同一轮次内不同生成次数的文本质量比较（支持任意gen_num）
    # 3. 训练目标：让模型偏好高质量文本，避免低质量文本
    # 4. 使用LoRA进行高效微调
    
    if [ ! -d "$adapter_output_dir" ]; then
        pushd "$llama_factory_dir" > /dev/null
        
        # 添加训练日志文件
        local log_file="${adapter_output_dir}/training.log"
        mkdir -p "$adapter_output_dir"
        
        echo "开始DPO训练，日志保存到: $log_file"
        
        llamafactory-cli train \
            --model_name_or_path $llm_path \
            --stage dpo \
            --do_train \
            --finetuning_type lora \
            --lora_rank $lora_rank \
            --lora_target all \
            --pref_beta 0.1 \
            --pref_loss sigmoid \
            --dataset $dataset_name \
            --dataset_dir "/home/user2/projects/TeR_TSF/llama-factory-main/data" \
            --template $llm_template \
            --cutoff_len 5120 \
            --max_samples 3000 \
            --overwrite_cache \
            --preprocessing_num_workers 128 \
            --dataloader_num_workers 128 \
            --output_dir $adapter_output_dir \
            --logging_steps 10 \
            --save_steps 500 \
            --per_device_train_batch_size $per_device_train_batch_size \
            --gradient_accumulation_steps 8 \
            --learning_rate $lr \
            --num_train_epochs $num_train_epochs \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --bf16 \
            --ddp_timeout 180000000 \
            2>&1 | tee "$log_file"
        
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            echo "DPO训练成功完成"
            echo "训练日志已保存到: $log_file"
        else
            echo "DPO训练失败，退出码: $exit_code"
            echo "请查看训练日志: $log_file"
            exit 1
        fi
        
        popd > /dev/null
    else
        echo "训练目录已存在，跳过训练..."
        echo "如需重新训练，请删除目录: $adapter_output_dir"
    fi
}

# 模型导出阶段
export_model() {
    local iter_idx=$1
    local llm_path=$2
    local llm_template=$3
    
    echo -e "\n===== 第 ${iter_idx} 轮导出 ====="
    
    if [ ! -d "$merge_output_dir" ]; then
        pushd "$llama_factory_dir" > /dev/null
        llamafactory-cli export \
            --model_name_or_path $llm_path \
            --adapter_name_or_path $adapter_output_dir \
            --template $llm_template \
            --export_dir $merge_output_dir \
            --export_size 5 \
            --export_device cpu \
        || { echo "导出失败！"; exit 1; }
        popd > /dev/null
    else
        echo "导出目录已存在，跳过导出..."
    fi
}

# 性能评估阶段
run_evaluation() {
    local iter_idx=$1
    local merge_output_dir=$2
    
    echo -e "\n===== 第 ${iter_idx} 轮评测 ====="
    
    # 性能评估阶段功能：
    # 1. 评估固定基准线：原始文本 + 未训练模型文本
    # 2. 评估训练效果：当前轮训练后模型文本
    # 3. 对比不同文本类型的预测性能
    # 4. 记录实验结果用于分析
    
    python evaluate.py \
        --data_dir /data2/user2/ter_tsf \
        --data_name $data_name \
        --llm_type $llm_type \
        --tsf_type $tsf_type \
        --hist_len $hist_len \
        --pred_len $pred_len \
        --batch_size $batch_size \
        --iter_idx $iter_idx \
        --llm_path $merge_output_dir \
        --exp_time $exp_time \
        --gen_num $gen_num \
        --dpo_lr $lr \
        --dpo_epoch $num_train_epochs \
        --down_sample $down_sample
}

# 主函数
main() {
    echo "=== TeR-TSF 训练流程开始 ==="
    echo "数据集: $data_name"
    echo "LLM类型: $llm_type"
    echo "TSF类型: $tsf_type"
    echo "迭代次数: $iter_num"
    echo "实验时间: $exp_time"
    echo "================================"
    
    # 获取LLM配置
    local llm_config=$(get_llm_config "$llm_type")
    local llm_template=$(echo "$llm_config" | cut -d: -f1)
    local initial_path=$(echo "$llm_config" | cut -d: -f2)
    
    # 执行迭代训练
    for ((iter_idx=0; iter_idx<$iter_num; iter_idx++)); do
        echo -e "\n========== 开始第 ${iter_idx} 轮迭代 =========="
        
        # 设置模型路径
        setup_model_paths $iter_idx $llm_template $initial_path
        
        # 数据准备阶段
        if ! run_data_preparation $iter_idx $llm_path; then
            echo "第 ${iter_idx} 轮训练被跳过，继续下一轮..."
            continue
        fi
        
        # TextFusionHTS训练阶段
        run_tfhts_training $iter_idx
        
        # DPO训练阶段
        local dataset_name="${data_name}_h${hist_len}_p${pred_len}_${llm_type}_${tsf_type}_genNum${gen_num}_iter${iter_idx}_${exp_time}"
        run_dpo_training $iter_idx $llm_path $llm_template $dataset_name
        
        # 模型导出阶段
        export_model $iter_idx $llm_path $llm_template
        
        # 性能评估阶段
        run_evaluation $iter_idx $merge_output_dir
        
        echo -e "========== 第 ${iter_idx} 轮迭代完成 ==========\n"
    done
    
    echo "=== TeR-TSF 训练流程完成 ==="
}

# 解析命令行参数
parse_arguments "$@"

# 执行主函数
main