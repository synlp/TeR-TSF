"""
TeR-TSF 性能监控工具
用于收集各阶段的详细性能数据，不修改原有代码
"""

import time
import psutil
import torch
import json
import pandas as pd
import numpy as np
import os
import threading
import queue
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import GPUtil


class PerformanceMonitor:
    """性能监控器 - 独立运行，不侵入原有代码"""
    
    def __init__(self, exp_name: str, output_dir: str = "./performance_data"):
        self.exp_name = exp_name
        self.output_dir = output_dir
        self.stage_stats = []
        self.timeline_data = []
        self.monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化GPU监控
        self.gpu_available = torch.cuda.is_available()
        self.gpus = GPUtil.getGPUs() if GPUtil.getGPUs() else []
        
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
            
        print(f"性能监控器已初始化: {exp_name}")
        print(f"GPU可用: {self.gpu_available}, GPU数量: {len(self.gpus)}")
    
    @contextmanager
    def monitor_stage(self, stage_name: str, metadata: Optional[Dict] = None):
        """
        阶段性能监控上下文管理器
        
        Args:
            stage_name: 阶段名称
            metadata: 额外的元数据信息
        """
        print(f"\n=== 开始监控阶段: {stage_name} ===")
        
        # 记录开始状态
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()
        
        # 系统资源监控
        start_cpu_percent = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory()
        start_disk_io = psutil.disk_io_counters()
        
        # GPU监控
        gpu_start_stats = self._get_gpu_stats()
        
        if self.gpu_available:
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated()
            start_gpu_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        
        # 启动连续监控
        self._start_continuous_monitoring()
        
        try:
            yield self
        finally:
            # 停止连续监控
            self._stop_continuous_monitoring()
            
            # 记录结束状态
            end_time = time.time()
            end_timestamp = datetime.now().isoformat()
            duration = end_time - start_time
            
            # 系统资源统计
            end_cpu_percent = psutil.cpu_percent(interval=None)
            end_memory = psutil.virtual_memory()
            end_disk_io = psutil.disk_io_counters()
            
            # GPU统计
            gpu_end_stats = self._get_gpu_stats()
            
            # 计算性能指标
            stage_stats = {
                'stage_name': stage_name,
                'experiment_name': self.exp_name,
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
                'duration_seconds': duration,
                
                # CPU指标
                'cpu_usage_start_percent': start_cpu_percent,
                'cpu_usage_end_percent': end_cpu_percent,
                'cpu_usage_avg_percent': (start_cpu_percent + end_cpu_percent) / 2,
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                
                # 内存指标
                'memory_total_gb': start_memory.total / 1024**3,
                'memory_start_used_gb': start_memory.used / 1024**3,
                'memory_end_used_gb': end_memory.used / 1024**3,
                'memory_delta_gb': (end_memory.used - start_memory.used) / 1024**3,
                'memory_peak_used_gb': end_memory.used / 1024**3,
                'memory_start_percent': start_memory.percent,
                'memory_end_percent': end_memory.percent,
                
                # 磁盘IO指标
                'disk_read_bytes': end_disk_io.read_bytes - start_disk_io.read_bytes if start_disk_io and end_disk_io else 0,
                'disk_write_bytes': end_disk_io.write_bytes - start_disk_io.write_bytes if start_disk_io and end_disk_io else 0,
                'disk_read_count': end_disk_io.read_count - start_disk_io.read_count if start_disk_io and end_disk_io else 0,
                'disk_write_count': end_disk_io.write_count - start_disk_io.write_count if start_disk_io and end_disk_io else 0,
            }
            
            # GPU指标
            if self.gpu_available:
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated()
                end_gpu_reserved = torch.cuda.memory_reserved()
                peak_gpu_memory = torch.cuda.max_memory_allocated()
                peak_gpu_reserved = torch.cuda.max_memory_reserved()
                
                stage_stats.update({
                    'gpu_memory_start_mb': start_gpu_memory / 1024**2,
                    'gpu_memory_end_mb': end_gpu_memory / 1024**2,
                    'gpu_memory_peak_mb': peak_gpu_memory / 1024**2,
                    'gpu_reserved_peak_mb': peak_gpu_reserved / 1024**2,
                    'gpu_memory_delta_mb': (end_gpu_memory - start_gpu_memory) / 1024**2,
                })
            
            # GPU硬件统计
            for i, (start_gpu, end_gpu) in enumerate(zip(gpu_start_stats, gpu_end_stats)):
                stage_stats.update({
                    f'gpu_{i}_memory_start_mb': start_gpu.get('memory_used', 0),
                    f'gpu_{i}_memory_end_mb': end_gpu.get('memory_used', 0),
                    f'gpu_{i}_memory_total_mb': end_gpu.get('memory_total', 0),
                    f'gpu_{i}_utilization_start': start_gpu.get('load', 0),
                    f'gpu_{i}_utilization_end': end_gpu.get('load', 0),
                    f'gpu_{i}_temperature_start': start_gpu.get('temperature', 0),
                    f'gpu_{i}_temperature_end': end_gpu.get('temperature', 0),
                })
            
            # 添加元数据
            if metadata:
                stage_stats.update(metadata)
            
            self.stage_stats.append(stage_stats)
            
            # 输出阶段摘要
            print(f"阶段 {stage_name} 完成:")
            print(f"  - 耗时: {duration:.2f}秒")
            print(f"  - CPU使用率: {stage_stats['cpu_usage_avg_percent']:.1f}%")
            print(f"  - 内存变化: {stage_stats['memory_delta_gb']:.3f}GB")
            if self.gpu_available:
                print(f"  - GPU内存峰值: {stage_stats['gpu_memory_peak_mb']:.1f}MB")
            print("=" * 50)
    
    def _get_gpu_stats(self) -> List[Dict]:
        """获取GPU统计信息"""
        gpu_stats = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                stats = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # 转换为百分比
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'temperature': gpu.temperature,
                }
                gpu_stats.append(stats)
        except Exception as e:
            print(f"GPU统计获取失败: {e}")
        
        return gpu_stats
    
    def _start_continuous_monitoring(self):
        """启动连续监控线程"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _stop_continuous_monitoring(self):
        """停止连续监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # 收集连续监控数据
        timeline_data = []
        while not self.data_queue.empty():
            try:
                timeline_data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        
        self.timeline_data.extend(timeline_data)
    
    def _monitor_resources(self):
        """资源监控线程函数"""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # CPU和内存监控
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                data_point = {
                    'timestamp': timestamp,
                    'datetime': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_used_gb': memory.used / 1024**3,
                    'memory_percent': memory.percent,
                }
                
                # GPU监控
                if self.gpu_available:
                    try:
                        data_point.update({
                            'torch_gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                            'torch_gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                        })
                    except Exception:
                        pass
                
                # 硬件GPU监控
                gpu_stats = self._get_gpu_stats()
                for i, gpu_stat in enumerate(gpu_stats):
                    data_point.update({
                        f'gpu_{i}_load': gpu_stat['load'],
                        f'gpu_{i}_memory_used_mb': gpu_stat['memory_used'],
                        f'gpu_{i}_temperature': gpu_stat['temperature'],
                    })
                
                self.data_queue.put(data_point)
                
            except Exception as e:
                print(f"监控线程错误: {e}")
            
            time.sleep(1.0)  # 每秒监控一次
    
    def save_results(self) -> tuple:
        """保存性能分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存阶段统计
        stage_df = pd.DataFrame(self.stage_stats)
        stage_file = os.path.join(self.output_dir, f"{self.exp_name}_stage_stats_{timestamp}.csv")
        stage_df.to_csv(stage_file, index=False)
        
        # 保存连续监控数据
        timeline_file = None
        if self.timeline_data:
            timeline_df = pd.DataFrame(self.timeline_data)
            timeline_file = os.path.join(self.output_dir, f"{self.exp_name}_timeline_{timestamp}.csv")
            timeline_df.to_csv(timeline_file, index=False)
        
        # 生成汇总报告
        summary_file = self._generate_summary_report(timestamp)
        
        print(f"\n性能分析结果已保存:")
        print(f"  - 阶段统计: {stage_file}")
        if timeline_file:
            print(f"  - 时间线数据: {timeline_file}")
        print(f"  - 汇总报告: {summary_file}")
        
        return stage_file, timeline_file, summary_file
    
    def _generate_summary_report(self, timestamp: str) -> str:
        """生成汇总报告"""
        if not self.stage_stats:
            return ""
        
        df = pd.DataFrame(self.stage_stats)
        
        # 计算汇总统计
        total_duration = df['duration_seconds'].sum()
        avg_cpu_usage = df['cpu_usage_avg_percent'].mean()
        peak_memory_gb = df['memory_peak_used_gb'].max()
        total_memory_delta = df['memory_delta_gb'].sum()
        
        # GPU统计
        gpu_stats = {}
        if 'gpu_memory_peak_mb' in df.columns:
            gpu_stats = {
                'peak_gpu_memory_mb': df['gpu_memory_peak_mb'].max(),
                'total_gpu_memory_delta_mb': df['gpu_memory_delta_mb'].sum(),
            }
        
        report = {
            'experiment_info': {
                'name': self.exp_name,
                'timestamp': timestamp,
                'total_stages': len(df),
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
            },
            'performance_summary': {
                'avg_cpu_usage_percent': avg_cpu_usage,
                'peak_memory_usage_gb': peak_memory_gb,
                'total_memory_delta_gb': total_memory_delta,
                **gpu_stats
            },
            'stage_breakdown': {
                'durations': df.groupby('stage_name')['duration_seconds'].agg(['sum', 'mean', 'std']).to_dict(),
                'memory_usage': df.groupby('stage_name')['memory_peak_used_gb'].agg(['max', 'mean', 'std']).to_dict(),
                'cpu_usage': df.groupby('stage_name')['cpu_usage_avg_percent'].agg(['mean', 'std']).to_dict(),
            },
            'bottleneck_analysis': {
                'slowest_stage': df.loc[df['duration_seconds'].idxmax(), 'stage_name'],
                'most_memory_intensive_stage': df.loc[df['memory_peak_used_gb'].idxmax(), 'stage_name'],
                'highest_cpu_usage_stage': df.loc[df['cpu_usage_avg_percent'].idxmax(), 'stage_name'],
            },
            'efficiency_metrics': {
                'time_per_stage_seconds': total_duration / len(df),
                'memory_efficiency_gb_per_second': peak_memory_gb / total_duration if total_duration > 0 else 0,
                'cpu_efficiency_percent_per_second': avg_cpu_usage / total_duration if total_duration > 0 else 0,
            }
        }
        
        # 保存报告
        report_file = os.path.join(self.output_dir, f"{self.exp_name}_summary_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_file


def create_stage_monitor(exp_name: str, stage_name: str, metadata: Optional[Dict] = None) -> PerformanceMonitor:
    """
    创建阶段性能监控器的便捷函数
    
    Args:
        exp_name: 实验名称
        stage_name: 阶段名称
        metadata: 元数据
    
    Returns:
        PerformanceMonitor实例
    """
    monitor = PerformanceMonitor(exp_name)
    return monitor


if __name__ == "__main__":
    # 测试性能监控器
    import time
    
    monitor = PerformanceMonitor("test_experiment")
    
    with monitor.monitor_stage("test_stage", {"batch_size": 32, "model": "test"}):
        # 模拟一些计算
        time.sleep(2)
        
        # 模拟内存使用
        data = [i for i in range(1000000)]
        time.sleep(1)
        
        del data
        time.sleep(1)
    
    # 保存结果
    monitor.save_results() 