# TeR-TSF 性能分析实验摘要

## 实验基本信息
- **实验时间**: 2025-09-23T23:42:02+08:00
- **数据集**: Energy
- **LLM类型**: qwen3-1.7b
- **TSF类型**: tfhts
- **配置**: hist_len=96, pred_len=12, batch_size=64, gen_num=2

## 实验结果

### 分阶段性能分析
- **执行状态**: true
- **执行时间**: 910秒
- **输出目录**: ./performance_experiments_20250923_211608/stage_analysis

### 参数敏感性分析
- **执行状态**: true
- **执行时间**: 7838秒
- **输出目录**: ./performance_experiments_20250923_211608/param_sensitivity

## 输出文件结构
```
./performance_experiments_20250923_211608/
├── experiment_config.json          # 实验配置
├── system_info.txt                 # 系统信息
├── logs/                          # 执行日志
│   ├── stage_analysis.log
│   └── param_sensitivity.log
├── stage_analysis/                # 分阶段分析结果
│   ├── performance_data/
│   └── *_comprehensive_report.json
├── param_sensitivity/             # 参数敏感性分析结果
│   ├── performance_data/
│   ├── logs/
│   ├── data/
│   └── *_comprehensive_report.json
└── summary/                       # 实验摘要
    ├── results.txt
    └── experiment_summary.md
```

## 下一步建议
1. 查看分阶段分析报告，识别性能瓶颈
2. 分析参数敏感性结果，优化关键参数
3. 使用生成的CSV数据进行可视化分析
4. 根据结果调整模型配置和训练策略
