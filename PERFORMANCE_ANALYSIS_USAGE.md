# TeR-TSF 性能分析实验使用指南

## 环境准备

```bash
# 激活conda环境
conda activate ter_tsf

# 验证依赖包
python -c "import psutil, torch, pandas, numpy, GPUtil; print('依赖包检查通过')"
```

## 快速开始

### 1. 运行完整的性能分析实验（带实时监控）

```bash
# 使用默认配置运行完整实验
./run_performance_experiments.sh

# 使用自定义配置
./run_performance_experiments.sh \
    --data_name Energy \
    --llm_type qwen3-1.7b \
    --tsf_type tfhts \
    --hist_len 96 \
    --pred_len 12 \
    --batch_size 64 \
    --gen_num 2
```

**实时监控功能**：
- ✅ **实时状态显示**：CPU、内存、GPU使用率
- ✅ **卡死检测**：5分钟无日志更新自动警告
- ✅ **进程管理**：支持Ctrl+C优雅终止
- ✅ **详细日志**：关键进展信息实时显示

### 2. 仅运行分阶段性能分析

```bash
# 单独运行分阶段分析
./run_performance_experiments.sh --stage_only \
    --data_name Agriculture \
    --hist_len 36 \
    --pred_len 6

# 或直接调用Python脚本
python stage_performance_analysis.py \
    --data_name Agriculture \
    --llm_type qwen3-1.7b \
    --tsf_type tfhts \
    --hist_len 36 \
    --pred_len 6 \
    --batch_size 64 \
    --gen_num 2
```

### 3. 仅运行参数敏感性分析

```bash
# 单独运行参数敏感性分析
./run_performance_experiments.sh --param_only \
    --max_workers 2

# 或直接调用Python脚本
python parameter_sensitivity_analysis.py \
    --data_name Energy \
    --llm_type qwen3-1.7b \
    --test_gen_num 2 4 8 \
    --test_batch_size 16 32 64 \
    --test_hist_len 36 96 192 \
    --test_pred_len 6 12 24 \
    --max_workers 2
```

### 4. 仅运行效率测试实验

```bash
# 测试Energy数据集在预测长度12,24,48下的运行时间
./run_performance_experiments.sh --efficiency_only --llm_type qwen3-1.7b

# 使用不同LLM类型进行效率测试
./run_performance_experiments.sh --efficiency_only --llm_type llama3.2-3b

# 或直接调用Python脚本
python efficiency_experiment.py \
    --llm_type qwen3-1.7b \
    --tsf_type tfhts \
    --hist_len 96 \
    --gen_num 2
```

## 脚本说明

### 1. 性能监控工具 (`performance_monitor.py`)
- **功能**: 提供底层的性能监控能力
- **特点**: 独立运行，不侵入原有代码
- **监控指标**: CPU、内存、GPU使用情况，磁盘IO等

### 2. 分阶段性能分析 (`stage_performance_analysis.py`)
- **功能**: 分析TeR-TSF框架各阶段的性能表现
- **包含阶段**: 
  - 数据准备阶段
  - TextFusionHTS训练阶段
  - DPO训练阶段
  - 评估阶段
- **输出**: 各阶段耗时、内存使用、CPU使用等详细指标

### 3. 参数敏感性分析 (`parameter_sensitivity_analysis.py`)
- **功能**: 分析不同参数配置对系统性能的影响
- **测试参数**: gen_num, batch_size, hist_len, pred_len
- **输出**: 参数影响排序、最优配置建议、可扩展性分析

### 4. 批量实验执行脚本 (`run_performance_experiments.sh`)
- **功能**: 自动化执行完整的性能分析流程
- **特点**: 
  - 支持并行执行
  - 自动依赖检查
  - 详细的日志记录
  - 实验结果汇总

### 5. 结果分析工具 (`analyze_performance_results.py`)
- **功能**: 处理和汇总实验结果
- **输出**: 论文用的结构化数据表格

### 6. 实时监控工具 (`real_time_monitor.py`, `monitor_experiment.py`)
- **功能**: 长时间运行实验的实时监控
- **特点**: 
  - 实时显示系统资源使用情况
  - 自动检测进程卡死状态
  - 支持优雅的进程终止
  - 生成详细的监控报告

### 7. 效率测试工具 (`efficiency_experiment.py`)
- **功能**: 专门的模型效率测试实验
- **特点**:
  - 固定测试Energy数据集
  - 测试预测长度: 12, 24, 48
  - 仅测量数据准备和DPO训练阶段
  - 固定配置: batch_size=64, max_batches=1, iteration=1, epoch=1

## 参数配置说明

### 基础配置参数
- `--data_name`: 数据集名称 (Energy, Agriculture, Economy, Health等)
- `--llm_type`: LLM类型 (qwen3-1.7b, llama3.2-1b, llama3.2-3b等)
- `--tsf_type`: TSF类型 (tfhts, mcd-tsf, time-llm)
- `--hist_len`: 历史序列长度 (36, 96, 192, 336)
- `--pred_len`: 预测序列长度 (6, 12, 24, 48)
- `--batch_size`: 批处理大小 (16, 32, 64, 128)
- `--gen_num`: 生成文本数量 (2, 4, 8, 16)

### 实验配置参数
- `--max_workers`: 最大并行线程数 (默认: 1)
- `--timeout`: 单个实验超时时间秒数 (默认: 3600)
- `--output_dir`: 输出目录
- `--stage_only`: 仅运行分阶段分析
- `--param_only`: 仅运行参数敏感性分析

## 输出结果结构

```
performance_experiments_YYYYMMDD_HHMMSS/
├── experiment_config.json          # 实验配置
├── system_info.txt                 # 系统信息
├── logs/                          # 执行日志
│   ├── stage_analysis.log
│   └── param_sensitivity.log
├── monitoring/                    # 实时监控数据 🆕
│   ├── stage_analysis_monitoring.json     # 阶段分析监控数据
│   ├── stage_analysis_final_report.json  # 阶段分析最终报告
│   ├── param_sensitivity_monitoring.json # 参数分析监控数据
│   ├── param_sensitivity_final_report.json # 参数分析最终报告
│   ├── stage_analysis_result.json        # 执行结果
│   └── param_sensitivity_result.json     # 执行结果
├── stage_analysis/                # 分阶段分析结果
│   ├── performance_data/          # 性能监控数据
│   │   ├── *_stage_stats_*.csv    # 阶段统计数据
│   │   ├── *_timeline_*.csv       # 时间线数据
│   │   └── *_summary_*.json       # 汇总报告
│   └── *_comprehensive_report.json # 综合报告
├── param_sensitivity/             # 参数敏感性分析结果
│   ├── performance_data/          # 性能监控数据
│   ├── logs/                      # 各实验日志
│   ├── data/                      # 实验数据
│   ├── analysis_results_*.csv     # 结构化分析结果
│   └── comprehensive_report_*.json # 综合报告
└── summary/                       # 实验摘要
    ├── results.txt                # 结果摘要
    └── experiment_summary.md      # Markdown格式摘要
```

## 数据分析

### 生成论文用表格

```bash
# 分析实验结果并生成论文用表格
python analyze_performance_results.py \
    --results_dir ./performance_experiments_YYYYMMDD_HHMMSS \
    --output_dir ./analysis_output
```

### 主要输出表格
1. **Table1_Stage_Performance.csv**: 各阶段性能汇总
2. **Table2_Parameter_Impact.csv**: 参数影响排序
3. **Table3_Optimal_Configurations.csv**: 最优配置建议

## 使用示例

### 示例1: 快速性能评估
```bash
# 使用默认配置快速评估Energy数据集
./run_performance_experiments.sh --data_name Energy
```

### 示例2: 详细参数敏感性分析
```bash
# 对Agriculture数据集进行详细的参数敏感性分析
./run_performance_experiments.sh \
    --data_name Agriculture \
    --hist_len 36 \
    --pred_len 6 \
    --param_only \
    --max_workers 4 \
    --timeout 1800
```

### 示例3: 多数据集对比分析
```bash
# 分别分析不同数据集的性能特征
for dataset in Energy Agriculture Economy Health; do
    ./run_performance_experiments.sh \
        --data_name $dataset \
        --stage_only \
        --output_dir ./results_${dataset}
done
```

### 示例4: 独立监控长时间实验
```bash
# 监控任意长时间运行的命令
python monitor_experiment.py python prepare_stage.py --data_name Energy \
    --timeout 3600 \
    --log-file ./my_experiment.log \
    --process-name data_preparation

# 监控DPO训练过程
python monitor_experiment.py llamafactory-cli train --config dpo_config.yaml \
    --timeout 7200 \
    --process-name dpo_training
```

## 故障排除

### 常见问题

1. **依赖包缺失**
   ```bash
   pip install psutil GPUtil pandas numpy torch
   ```

2. **GPU监控失败**
   - 确保安装了NVIDIA驱动和nvidia-smi
   - 检查GPUtil包是否正确安装

3. **内存不足**
   - 减小batch_size参数
   - 使用--down_sample参数进行数据下采样

4. **实验超时**
   - 增加--timeout参数值
   - 减少测试的参数范围

5. **进程卡死检测**
   - 监控会在5分钟无日志更新时发出警告
   - 检查 `monitoring/` 目录下的报告文件
   - 使用 `Ctrl+C` 优雅终止卡死的进程

6. **监控数据查看**
   ```bash
   # 查看实时监控数据
   cat monitoring/*_monitoring.json | jq '.cpu_usage[-10:]'
   
   # 查看最终报告
   cat monitoring/*_final_report.json | jq '.'
   ```

7. **独立监控使用**
   ```bash
   # 监控任意命令
   python monitor_experiment.py your_command --timeout 1800
   ```

### 调试模式

```bash
# 启用详细日志输出
python stage_performance_analysis.py --data_name Energy --verbose

# 使用小规模配置进行测试
./run_performance_experiments.sh \
    --data_name Energy \
    --batch_size 16 \
    --gen_num 2 \
    --timeout 600
```

## 注意事项

1. **资源要求**: 确保有足够的GPU内存和磁盘空间
2. **时间估算**: 完整实验可能需要几小时到十几小时
3. **数据路径**: 确保原始数据文件存在于正确路径
4. **并发控制**: 根据系统资源调整max_workers参数
5. **结果备份**: 及时备份重要的实验结果

## 技术支持

如遇到问题，请检查：
1. 日志文件中的详细错误信息
2. 系统资源使用情况
3. 数据文件完整性
4. 环境配置正确性 