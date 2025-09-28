#!/usr/bin/env python3
"""
TeR-TSF 独立实验监控脚本
用于监控长时间运行的实验，提供实时状态和卡死检测
"""

import argparse
import sys
import os
import signal
from real_time_monitor import run_with_monitoring


def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\n收到信号 {signum}，正在优雅退出...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="TeR-TSF 实验监控工具")
    parser.add_argument("command", nargs="+", help="要监控的命令")
    parser.add_argument("--log-file", type=str, help="日志文件路径")
    parser.add_argument("--process-name", type=str, help="进程名称")
    parser.add_argument("--timeout", type=int, default=3600, help="超时时间（秒）")
    parser.add_argument("--no-log-tail", action="store_true", help="禁用日志尾部监控")
    parser.add_argument("--monitor-dir", type=str, default="./monitoring", help="监控输出目录")
    
    args = parser.parse_args()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 确定进程名称和日志文件
    process_name = args.process_name or os.path.basename(args.command[0])
    log_file = args.log_file or f"{process_name}_monitor.log"
    
    print(f"🔍 TeR-TSF 实验监控工具")
    print(f"   监控命令: {' '.join(args.command)}")
    print(f"   进程名称: {process_name}")
    print(f"   日志文件: {log_file}")
    print(f"   超时设置: {args.timeout}秒")
    print(f"   监控目录: {args.monitor_dir}")
    print(f"   日志监控: {'禁用' if args.no_log_tail else '启用'}")
    print("   " + "="*50)
    
    # 执行监控
    try:
        result = run_with_monitoring(
            command=args.command,
            log_file=log_file,
            process_name=process_name,
            timeout=args.timeout,
            enable_log_tail=not args.no_log_tail
        )
        
        print(f"\n📋 监控结果摘要:")
        print(f"   成功: {result['success']}")
        print(f"   返回码: {result['return_code']}")
        print(f"   运行时间: {result['duration']:.1f}秒")
        
        if result.get('timeout_occurred'):
            print(f"   ⏰ 超时终止")
        if result.get('manually_killed'):
            print(f"   ⌨️  手动终止")
        if result.get('error'):
            print(f"   ❌ 错误: {result['error']}")
        
        # 返回适当的退出码
        sys.exit(0 if result['success'] else 1)
        
    except KeyboardInterrupt:
        print(f"\n⌨️  用户中断监控")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 监控过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 