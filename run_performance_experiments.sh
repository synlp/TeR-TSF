#!/bin/bash

# TeR-TSF 性能分析实验批量执行脚本
# 用于自动化运行分阶段性能分析和参数敏感性分析

set -e  # 遇到错误立即退出
export CUDA_VISIBLE_DEVICES=1
# 颜色输出函数
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# 默认配置
DATA_NAME="Energy"
LLM_TYPE="qwen3-1.7b"
TSF_TYPE="tfhts"
HIST_LEN=96
PRED_LEN=12
BATCH_SIZE=64
GEN_NUM=2
OUTPUT_BASE_DIR="./performance_experiments_$(date +%Y%m%d_%H%M%S)"

# 实验类型选择
RUN_STAGE_ANALYSIS=true
RUN_PARAM_SENSITIVITY=true
RUN_EFFICIENCY_TEST=false
MAX_WORKERS=1
# TIMEOUT参数已移除，使用实时输出模式

# 参数解析函数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --data_name) DATA_NAME="$2"; shift ;;
            --llm_type) LLM_TYPE="$2"; shift ;;
            --tsf_type) TSF_TYPE="$2"; shift ;;
            --hist_len) HIST_LEN="$2"; shift ;;
            --pred_len) PRED_LEN="$2"; shift ;;
            --batch_size) BATCH_SIZE="$2"; shift ;;
            --gen_num) GEN_NUM="$2"; shift ;;
            --output_dir) OUTPUT_BASE_DIR="$2"; shift ;;
            --max_workers) MAX_WORKERS="$2"; shift ;;
            # --timeout参数已移除
            --stage_only) RUN_STAGE_ANALYSIS=true; RUN_PARAM_SENSITIVITY=false; RUN_EFFICIENCY_TEST=false ;;
            --param_only) RUN_STAGE_ANALYSIS=false; RUN_PARAM_SENSITIVITY=true; RUN_EFFICIENCY_TEST=false ;;
            --efficiency_only) RUN_STAGE_ANALYSIS=false; RUN_PARAM_SENSITIVITY=false; RUN_EFFICIENCY_TEST=true ;;
            --help) show_help; exit 0 ;;
            *) print_error "未知参数: $1"; show_help; exit 1 ;;
        esac
        shift
    done
}

# 帮助信息
show_help() {
    cat << EOF
TeR-TSF 性能分析实验批量执行脚本

用法: $0 [选项]

基础配置选项:
    --data_name <name>      数据集名称 (默认: Energy)
    --llm_type <type>       LLM类型 (默认: qwen3-1.7b)
    --tsf_type <type>       TSF类型 (默认: tfhts)
    --hist_len <len>        历史序列长度 (默认: 96)
    --pred_len <len>        预测序列长度 (默认: 12)
    --batch_size <size>     批处理大小 (默认: 64)
    --gen_num <num>         生成文本数量 (默认: 2)

实验配置选项:
    --output_dir <dir>      输出目录 (默认: ./performance_experiments_时间戳)
    --max_workers <num>     最大并行工作线程数 (默认: 1)
    # 超时参数已移除，现在使用实时输出模式
    --stage_only           仅运行分阶段性能分析
    --param_only           仅运行参数敏感性分析
    --efficiency_only      仅运行效率测试实验
    --help                 显示此帮助信息

示例:
    # 运行完整的性能分析实验
    $0 --data_name Energy --llm_type qwen3-1.7b

    # 仅运行分阶段性能分析
    $0 --stage_only --data_name Agriculture --hist_len 36 --pred_len 6

    # 运行参数敏感性分析，使用4个并行线程
    $0 --param_only --max_workers 4

    # 仅运行效率测试实验
    $0 --efficiency_only --data_name Energy --llm_type qwen3-1.7b
EOF
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python脚本
    local required_scripts=("performance_monitor.py")
    
    if [[ "$RUN_STAGE_ANALYSIS" == "true" ]]; then
        required_scripts+=("stage_performance_analysis.py")
    fi
    
    if [[ "$RUN_PARAM_SENSITIVITY" == "true" ]]; then
        required_scripts+=("parameter_sensitivity_analysis.py")
    fi
    
    if [[ "$RUN_EFFICIENCY_TEST" == "true" ]]; then
        required_scripts+=("efficiency_experiment.py")
    fi
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            print_error "缺少必需的脚本: $script"
            exit 1
        fi
    done
    
    # 检查Python依赖
    python -c "import psutil, torch, pandas, numpy, GPUtil" 2>/dev/null || {
        print_error "缺少Python依赖包，请安装: psutil torch pandas numpy GPUtil"
        exit 1
    }
    
    # 检查原始脚本
    if [[ ! -f "prepare_stage.py" ]]; then
        print_warning "未找到 prepare_stage.py，分阶段分析可能失败"
    fi
    
    if [[ ! -f "evaluate.py" ]]; then
        print_warning "未找到 evaluate.py，评估阶段分析可能失败"
    fi
    
    print_success "依赖检查完成"
}

# 创建输出目录结构
setup_output_directories() {
    print_info "创建输出目录结构..."
    
    mkdir -p "$OUTPUT_BASE_DIR"
    mkdir -p "$OUTPUT_BASE_DIR/stage_analysis"
    mkdir -p "$OUTPUT_BASE_DIR/param_sensitivity"
    mkdir -p "$OUTPUT_BASE_DIR/logs"
    mkdir -p "$OUTPUT_BASE_DIR/summary"
    
    # 保存实验配置
    cat > "$OUTPUT_BASE_DIR/experiment_config.json" << EOF
{
    "experiment_timestamp": "$(date -Iseconds)",
    "base_config": {
        "data_name": "$DATA_NAME",
        "llm_type": "$LLM_TYPE",
        "tsf_type": "$TSF_TYPE",
        "hist_len": $HIST_LEN,
        "pred_len": $PRED_LEN,
        "batch_size": $BATCH_SIZE,
        "gen_num": $GEN_NUM
    },
    "experiment_settings": {
        "run_stage_analysis": $RUN_STAGE_ANALYSIS,
        "run_param_sensitivity": $RUN_PARAM_SENSITIVITY,
        "run_efficiency_test": $RUN_EFFICIENCY_TEST,
        "max_workers": $MAX_WORKERS,
        "realtime_output": true,
        "output_base_dir": "$OUTPUT_BASE_DIR"
    }
}
EOF
    
    print_success "输出目录结构创建完成: $OUTPUT_BASE_DIR"
}

# 记录系统信息
record_system_info() {
    print_info "记录系统信息..."
    
    local system_info_file="$OUTPUT_BASE_DIR/system_info.txt"
    
    cat > "$system_info_file" << EOF
=== 系统信息 ===
时间戳: $(date -Iseconds)
主机名: $(hostname)
用户: $(whoami)
工作目录: $(pwd)

=== 硬件信息 ===
CPU信息:
$(lscpu | head -20)

内存信息:
$(free -h)

磁盘信息:
$(df -h | head -10)

=== GPU信息 ===
EOF
    
    # 添加GPU信息
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU信息:" >> "$system_info_file"
        nvidia-smi >> "$system_info_file" 2>&1
    else
        echo "未检测到NVIDIA GPU" >> "$system_info_file"
    fi
    
    # 添加Python环境信息
    cat >> "$system_info_file" << EOF

=== Python环境信息 ===
Python版本: $(python --version)
PyTorch版本: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")
CUDA版本: $(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "未安装")

=== 相关进程 ===
$(ps aux | grep -E "(python|jupyter)" | head -10)
EOF
    
    print_success "系统信息已记录: $system_info_file"
}

# 运行分阶段性能分析
run_stage_analysis() {
    if [[ "$RUN_STAGE_ANALYSIS" != "true" ]]; then
        return 0
    fi
    
    print_info "开始分阶段性能分析..."
    
    local stage_output_dir="$OUTPUT_BASE_DIR/stage_analysis"
    local log_file="$OUTPUT_BASE_DIR/logs/stage_analysis.log"
    local monitor_dir="$OUTPUT_BASE_DIR/monitoring"
    
    # 创建监控目录
    mkdir -p "$monitor_dir"
    
    # 构建命令
    local cmd="python stage_performance_analysis.py \
        --data_name $DATA_NAME \
        --llm_type $LLM_TYPE \
        --tsf_type $TSF_TYPE \
        --hist_len $HIST_LEN \
        --pred_len $PRED_LEN \
        --batch_size $BATCH_SIZE \
        --gen_num $GEN_NUM \
        --output_dir $stage_output_dir"
    
    print_info "执行命令: $cmd"
    print_info "启用实时监控，日志文件: $log_file"
    print_info "监控数据将保存到: $monitor_dir"
    
    # 直接执行Python脚本，实时显示输出
    local start_time=$(date +%s)
    
    print_info "开始执行分阶段性能分析..."
    echo "================================================================================"
    
    # 使用tee同时输出到终端和日志文件
    eval "$cmd" 2>&1 | tee "$log_file"
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "================================================================================"
    
    # 创建结果JSON文件
    cat > "$monitor_dir/stage_analysis_result.json" << EOF
{
  "success": $([ $exit_code -eq 0 ] && echo "true" || echo "false"),
  "return_code": $exit_code,
  "duration": $duration,
  "timeout_occurred": false,
  "manually_killed": false
}
EOF
    
    if [ $exit_code -eq 0 ]; then
        print_success "分阶段性能分析完成，耗时: ${duration}秒"
        
        # 记录结果摘要
        echo "stage_analysis_success: true" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "stage_analysis_duration: ${duration}" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        
        return 0
    else
        print_error "分阶段性能分析失败，耗时: ${duration}秒"
        print_error "详细错误信息请查看: $log_file"
        print_error "监控报告请查看: $monitor_dir/"
        
        # 记录失败结果
        echo "stage_analysis_success: false" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "stage_analysis_duration: ${duration}" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "stage_analysis_error: execution failed" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        
        return 1
    fi
}

# 运行参数敏感性分析
run_param_sensitivity() {
    if [[ "$RUN_PARAM_SENSITIVITY" != "true" ]]; then
        return 0
    fi
    
    print_info "开始参数敏感性分析..."
    
    local param_output_dir="$OUTPUT_BASE_DIR/param_sensitivity"
    local log_file="$OUTPUT_BASE_DIR/logs/param_sensitivity.log"
    local monitor_dir="$OUTPUT_BASE_DIR/monitoring"
    
    # 创建监控目录
    mkdir -p "$monitor_dir"
    
    # 构建命令
    local cmd="python parameter_sensitivity_analysis.py \
        --data_name $DATA_NAME \
        --llm_type $LLM_TYPE \
        --tsf_type $TSF_TYPE \
        --hist_len $HIST_LEN \
        --pred_len $PRED_LEN \
        --batch_size $BATCH_SIZE \
        --gen_num $GEN_NUM \
        --output_dir $param_output_dir \
        --max_workers $MAX_WORKERS \
        --test_gen_num 2 4 8 \
        --test_batch_size 16 32 64 \
        --test_hist_len 36 96 192 \
        --test_pred_len 6 12 24"
    
    print_info "执行命令: $cmd"
    print_info "启用实时监控，日志文件: $log_file"
    print_info "监控数据将保存到: $monitor_dir"
    print_warning "参数敏感性分析可能需要很长时间，请耐心等待..."
    
    # 直接执行Python脚本，实时显示输出
    local start_time=$(date +%s)
    
    print_info "开始执行参数敏感性分析..."
    echo "================================================================================"
    
    # 使用tee同时输出到终端和日志文件
    eval "$cmd" 2>&1 | tee "$log_file"
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "================================================================================"
    
    # 创建结果JSON文件
    cat > "$monitor_dir/param_sensitivity_result.json" << EOF
{
  "success": $([ $exit_code -eq 0 ] && echo "true" || echo "false"),
  "return_code": $exit_code,
  "duration": $duration,
  "timeout_occurred": false,
  "manually_killed": false
}
EOF
    
    if [ $exit_code -eq 0 ]; then
        print_success "参数敏感性分析完成，耗时: ${duration}秒"
        
        # 记录结果摘要
        echo "param_sensitivity_success: true" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "param_sensitivity_duration: ${duration}" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        
        return 0
    else
        print_error "参数敏感性分析失败，耗时: ${duration}秒"
        print_error "详细错误信息请查看: $log_file"
        print_error "监控报告请查看: $monitor_dir/"
        
        # 记录失败结果
        echo "param_sensitivity_success: false" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "param_sensitivity_duration: ${duration}" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "param_sensitivity_error: execution failed" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        
        return 1
    fi
}

# 运行效率测试实验
run_efficiency_test() {
    if [[ "$RUN_EFFICIENCY_TEST" != "true" ]]; then
        return 0
    fi
    
    print_info "开始效率测试实验..."
    
    local efficiency_output_dir="$OUTPUT_BASE_DIR/efficiency_test"
    local log_file="$OUTPUT_BASE_DIR/logs/efficiency_test.log"
    local monitor_dir="$OUTPUT_BASE_DIR/monitoring"
    
    # 创建监控目录
    mkdir -p "$monitor_dir"
    
    # 构建命令
    local cmd="python efficiency_experiment.py \
        --llm_type $LLM_TYPE \
        --tsf_type $TSF_TYPE \
        --hist_len $HIST_LEN \
        --gen_num $GEN_NUM \
        --output_dir $efficiency_output_dir"
    
    print_info "执行命令: $cmd"
    print_info "启用实时监控，日志文件: $log_file"
    print_info "监控数据将保存到: $monitor_dir"
    print_warning "效率测试将测量Energy数据集在预测长度12,24,48下的运行时间..."
    
    # 直接执行Python脚本，实时显示输出
    local start_time=$(date +%s)
    
    print_info "开始执行效率测试实验..."
    echo "================================================================================"
    
    # 使用tee同时输出到终端和日志文件
    eval "$cmd" 2>&1 | tee "$log_file"
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "================================================================================"
    
    # 创建结果JSON文件
    cat > "$monitor_dir/efficiency_test_result.json" << EOF
{
  "success": $([ $exit_code -eq 0 ] && echo "true" || echo "false"),
  "return_code": $exit_code,
  "duration": $duration,
  "timeout_occurred": false,
  "manually_killed": false
}
EOF
    
    if [ $exit_code -eq 0 ]; then
        print_success "效率测试实验完成，耗时: ${duration}秒"
        
        # 记录结果摘要
        echo "efficiency_test_success: true" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "efficiency_test_duration: ${duration}" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        
        return 0
    else
        print_error "效率测试实验失败，耗时: ${duration}秒"
        print_error "详细错误信息请查看: $log_file"
        print_error "监控报告请查看: $monitor_dir/"
        
        # 记录失败结果
        echo "efficiency_test_success: false" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "efficiency_test_duration: ${duration}" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        echo "efficiency_test_error: execution failed" >> "$OUTPUT_BASE_DIR/summary/results.txt"
        
        return 1
    fi
}

# 生成实验摘要
generate_experiment_summary() {
    print_info "生成实验摘要..."
    
    local summary_file="$OUTPUT_BASE_DIR/summary/experiment_summary.md"
    local results_file="$OUTPUT_BASE_DIR/summary/results.txt"
    
    # 读取结果
    local stage_success="false"
    local param_success="false"
    local efficiency_success="false"
    local stage_duration="0"
    local param_duration="0"
    local efficiency_duration="0"
    
    if [[ -f "$results_file" ]]; then
        stage_success=$(grep "stage_analysis_success:" "$results_file" | cut -d' ' -f2 || echo "false")
        param_success=$(grep "param_sensitivity_success:" "$results_file" | cut -d' ' -f2 || echo "false")
        efficiency_success=$(grep "efficiency_test_success:" "$results_file" | cut -d' ' -f2 || echo "false")
        stage_duration=$(grep "stage_analysis_duration:" "$results_file" | cut -d' ' -f2 || echo "0")
        param_duration=$(grep "param_sensitivity_duration:" "$results_file" | cut -d' ' -f2 || echo "0")
        efficiency_duration=$(grep "efficiency_test_duration:" "$results_file" | cut -d' ' -f2 || echo "0")
    fi
    
    # 生成Markdown摘要
    cat > "$summary_file" << EOF
# TeR-TSF 性能分析实验摘要

## 实验基本信息
- **实验时间**: $(date -Iseconds)
- **数据集**: $DATA_NAME
- **LLM类型**: $LLM_TYPE
- **TSF类型**: $TSF_TYPE
- **配置**: hist_len=$HIST_LEN, pred_len=$PRED_LEN, batch_size=$BATCH_SIZE, gen_num=$GEN_NUM

## 实验结果

### 分阶段性能分析
- **执行状态**: $stage_success
- **执行时间**: ${stage_duration}秒
- **输出目录**: $OUTPUT_BASE_DIR/stage_analysis

### 参数敏感性分析
- **执行状态**: $param_success
- **执行时间**: ${param_duration}秒
- **输出目录**: $OUTPUT_BASE_DIR/param_sensitivity

### 效率测试实验
- **执行状态**: $efficiency_success
- **执行时间**: ${efficiency_duration}秒
- **输出目录**: $OUTPUT_BASE_DIR/efficiency_test

## 输出文件结构
\`\`\`
$OUTPUT_BASE_DIR/
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
├── efficiency_test/               # 效率测试实验结果
│   ├── performance_data/
│   ├── logs/
│   └── *_comprehensive_report.json
└── summary/                       # 实验摘要
    ├── results.txt
    └── experiment_summary.md
\`\`\`

## 下一步建议
EOF
    
    # 添加建议
    if [[ "$stage_success" == "true" && "$param_success" == "true" && "$efficiency_success" == "true" ]]; then
        cat >> "$summary_file" << EOF
1. 查看分阶段分析报告，识别性能瓶颈
2. 分析参数敏感性结果，优化关键参数
3. 查看效率测试报告，了解不同预测长度下的运行时间
4. 使用生成的CSV数据进行可视化分析
5. 根据结果调整模型配置和训练策略
EOF
    elif [[ "$efficiency_success" == "true" ]]; then
        cat >> "$summary_file" << EOF
1. 效率测试成功完成，可以查看Energy数据集在不同预测长度下的运行时间
2. 分析数据准备和DPO训练阶段的时间开销
3. 根据效率结果优化批处理和模型配置
EOF
    elif [[ "$stage_success" == "true" ]]; then
        cat >> "$summary_file" << EOF
1. 分阶段分析成功完成，可以查看性能瓶颈报告
2. 参数敏感性分析失败，建议检查日志并重新运行
3. 考虑减少参数范围或增加超时时间
EOF
    elif [[ "$param_success" == "true" ]]; then
        cat >> "$summary_file" << EOF
1. 参数敏感性分析成功完成，可以查看参数影响报告
2. 分阶段分析失败，建议检查环境和依赖
3. 可以使用 --stage_only 选项单独重新运行分阶段分析
EOF
    else
        cat >> "$summary_file" << EOF
1. 两个分析都失败了，请检查：
   - 系统环境和依赖是否正确安装
   - 数据路径是否存在
   - GPU资源是否充足
   - 网络连接是否正常
2. 查看详细日志文件排查问题
3. 考虑使用更小的配置参数进行测试
EOF
    fi
    
    print_success "实验摘要已生成: $summary_file"
}

# 主函数
main() {
    print_info "=== TeR-TSF 性能分析实验开始 ==="
    
    # 解析参数
    parse_arguments "$@"
    
    # 显示配置
    print_info "实验配置:"
    print_info "  数据集: $DATA_NAME"
    print_info "  LLM类型: $LLM_TYPE"
    print_info "  TSF类型: $TSF_TYPE"
    print_info "  序列配置: hist_len=$HIST_LEN, pred_len=$PRED_LEN"
    print_info "  批处理配置: batch_size=$BATCH_SIZE, gen_num=$GEN_NUM"
    print_info "  实验类型: 分阶段分析=$RUN_STAGE_ANALYSIS, 参数敏感性=$RUN_PARAM_SENSITIVITY, 效率测试=$RUN_EFFICIENCY_TEST"
    print_info "  并行配置: max_workers=$MAX_WORKERS, 实时输出模式"
    print_info "  输出目录: $OUTPUT_BASE_DIR"
    
    # 检查依赖
    check_dependencies
    
    # 设置输出目录
    setup_output_directories
    
    # 记录系统信息
    record_system_info
    
    # 创建结果文件
    touch "$OUTPUT_BASE_DIR/summary/results.txt"
    echo "experiment_start_time: $(date -Iseconds)" > "$OUTPUT_BASE_DIR/summary/results.txt"
    
    # 执行实验
    local overall_success=true
    
    if ! run_stage_analysis; then
        overall_success=false
    fi
    
    if ! run_param_sensitivity; then
        overall_success=false
    fi
    
    if ! run_efficiency_test; then
        overall_success=false
    fi
    
    # 记录结束时间
    echo "experiment_end_time: $(date -Iseconds)" >> "$OUTPUT_BASE_DIR/summary/results.txt"
    echo "overall_success: $overall_success" >> "$OUTPUT_BASE_DIR/summary/results.txt"
    
    # 生成摘要
    generate_experiment_summary
    
    # 最终结果
    if [[ "$overall_success" == "true" ]]; then
        print_success "=== 所有实验成功完成 ==="
        print_success "结果目录: $OUTPUT_BASE_DIR"
        print_success "摘要报告: $OUTPUT_BASE_DIR/summary/experiment_summary.md"
        exit 0
    else
        print_warning "=== 部分实验失败 ==="
        print_warning "请查看日志文件和摘要报告了解详情"
        print_warning "结果目录: $OUTPUT_BASE_DIR"
        exit 1
    fi
}

# 执行主函数
main "$@" 