#!/bin/bash

# GPU显存清理脚本
# 用于清理被kill的进程仍占用的GPU显存
# 使用方法: ./gpu_memory_cleanup.sh <PID>

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 检查参数
if [ $# -ne 1 ]; then
    print_error "使用方法: $0 <PID>"
    print_info "示例: $0 512626"
    exit 1
fi

TARGET_PID=$1

# 验证PID格式
if ! [[ "$TARGET_PID" =~ ^[0-9]+$ ]]; then
    print_error "PID必须是数字: $TARGET_PID"
    exit 1
fi

print_info "开始清理PID $TARGET_PID 占用的GPU显存..."

# 1. 检查进程是否还在运行
print_info "检查进程状态..."
if ps -p "$TARGET_PID" > /dev/null 2>&1; then
    print_warning "进程 $TARGET_PID 仍在运行，建议先正常终止该进程"
    read -p "是否继续清理？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "操作已取消"
        exit 0
    fi
else
    print_info "进程 $TARGET_PID 已不存在，检查GPU显存占用..."
fi

# 2. 检查GPU显存占用情况
print_info "检查GPU显存占用情况..."
GPU_USAGE=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^$TARGET_PID," || true)

if [ -z "$GPU_USAGE" ]; then
    print_info "未发现PID $TARGET_PID 占用GPU显存"
    exit 0
else
    print_warning "发现显存占用: $GPU_USAGE"
fi

# 3. 查找相关的CUDA UVM文件描述符
print_info "查找相关的CUDA文件描述符..."
CUDA_FDS=$(lsof 2>/dev/null | grep "cuda-uvmfd.*-$TARGET_PID@" | awk '{print $2}' | sort -u || true)

if [ -z "$CUDA_FDS" ]; then
    print_warning "未找到与PID $TARGET_PID 相关的CUDA文件描述符"
else
    print_info "找到相关的进程PID: $(echo $CUDA_FDS | tr '\n' ' ')"
fi

# 4. 查找使用GPU设备文件的进程
print_info "检查GPU设备文件使用情况..."
GPU_PROCESSES=""
for gpu_dev in /dev/nvidia*; do
    if [[ -c "$gpu_dev" ]]; then
        gpu_procs=$(fuser "$gpu_dev" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' || true)
        if [ -n "$gpu_procs" ]; then
            GPU_PROCESSES="$GPU_PROCESSES $gpu_procs"
        fi
    fi
done

# 5. 合并需要清理的进程列表
CLEANUP_PIDS=""
if [ -n "$CUDA_FDS" ]; then
    CLEANUP_PIDS="$CLEANUP_PIDS $CUDA_FDS"
fi
if [ -n "$GPU_PROCESSES" ]; then
    CLEANUP_PIDS="$CLEANUP_PIDS $GPU_PROCESSES"
fi

# 去重并排序
CLEANUP_PIDS=$(echo $CLEANUP_PIDS | tr ' ' '\n' | sort -u | grep -v "^$TARGET_PID$" | tr '\n' ' ' || true)

if [ -z "$CLEANUP_PIDS" ]; then
    print_warning "未找到需要清理的相关进程"
else
    print_info "准备清理以下进程: $CLEANUP_PIDS"
    
    # 显示进程信息
    print_info "相关进程信息:"
    for pid in $CLEANUP_PIDS; do
        if ps -p "$pid" > /dev/null 2>&1; then
            ps -p "$pid" -o pid,ppid,cmd --no-headers | head -1
        fi
    done
    
    # 确认清理
    read -p "确认清理这些进程？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "操作已取消"
        exit 0
    fi
    
    # 6. 清理进程
    print_info "正在清理相关进程..."
    for pid in $CLEANUP_PIDS; do
        if ps -p "$pid" > /dev/null 2>&1; then
            print_info "终止进程 $pid..."
            kill -9 "$pid" 2>/dev/null || true
            sleep 0.1
        fi
    done
fi

# 7. 等待清理完成
print_info "等待GPU显存释放..."
sleep 2

# 8. 验证清理结果
print_info "验证清理结果..."
FINAL_USAGE=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep "^$TARGET_PID," || true)

if [ -z "$FINAL_USAGE" ]; then
    print_success "GPU显存清理成功！PID $TARGET_PID 不再占用GPU显存"
else
    print_warning "清理后仍有显存占用: $FINAL_USAGE"
    print_info "可能需要重启系统或联系管理员重置GPU"
fi

# 9. 显示当前GPU状态
print_info "当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

print_info "清理操作完成" 