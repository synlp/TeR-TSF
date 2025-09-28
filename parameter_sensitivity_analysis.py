"""
TeR-TSF 参数敏感性分析实验
通过批量实验收集不同参数配置下的性能数据
用于分析各参数对系统性能的影响
"""

import subprocess
import os
import sys
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from performance_monitor import PerformanceMonitor


class ParameterSensitivityAnalyzer:
    """参数敏感性分析器"""
    
    def __init__(self, base_config: Dict, output_dir: str = "./param_sensitivity_results"):
        self.base_config = base_config
        self.output_dir = output_dir
        self.results = []
        self.lock = threading.Lock()
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "performance_data"), exist_ok=True)
        
        # 保存实验配置
        self.config_file = os.path.join(output_dir, "experiment_config.json")
        with open(self.config_file, 'w') as f:
            json.dump({
                'experiment_type': 'parameter_sensitivity_analysis',
                'timestamp': datetime.now().isoformat(),
                'base_config': base_config,
                'output_dir': output_dir
            }, f, indent=2)
        
        print(f"参数敏感性分析器已初始化")
        print(f"基础配置: {base_config}")
        print(f"输出目录: {output_dir}")
    
    def _run_command_with_realtime_output(self, cmd: List[str], stage_name: str, log_file: str) -> Dict:
        """运行命令并实时显示输出，同时保存到日志文件"""
        print(f"\n[{stage_name}] 执行命令: {' '.join(cmd)}")
        print(f"[{stage_name}] 开始时间: {datetime.now().isoformat()}")
        print(f"[{stage_name}] 日志文件: {log_file}")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # 使用Popen进行实时输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_lines = []
            stderr_lines = []
            
            def read_stdout():
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.rstrip()
                        stdout_lines.append(line)
                        print(f"[{stage_name}] {line}")
                        sys.stdout.flush()
                process.stdout.close()
            
            def read_stderr():
                for line in iter(process.stderr.readline, ''):
                    if line:
                        line = line.rstrip()
                        stderr_lines.append(line)
                        print(f"[{stage_name}] ERROR: {line}")
                        sys.stderr.flush()
                process.stderr.close()
            
            # 启动读取线程
            stdout_thread = threading.Thread(target=read_stdout)
            stderr_thread = threading.Thread(target=read_stderr)
            
            stdout_thread.start()
            stderr_thread.start()
            
            # 等待进程完成
            return_code = process.wait()
            
            # 等待读取线程完成
            stdout_thread.join()
            stderr_thread.join()
            
            end_time = time.time()
            
            # 写入日志文件
            with open(log_file, 'w') as f:
                f.write(f"执行命令: {' '.join(cmd)}\n")
                f.write(f"开始时间: {datetime.now().isoformat()}\n")
                f.write("="*80 + "\n")
                f.write(f"结束时间: {datetime.now().isoformat()}\n")
                f.write(f"执行时间: {end_time - start_time:.2f}秒\n")
                f.write(f"返回码: {return_code}\n")
                f.write(f"标准输出行数: {len(stdout_lines)}\n")
                f.write(f"标准错误行数: {len(stderr_lines)}\n")
                f.write("="*80 + "\n")
                if stdout_lines:
                    f.write("STDOUT:\n")
                    f.write("\n".join(stdout_lines) + "\n")
                f.write("="*80 + "\n")
                if stderr_lines:
                    f.write("STDERR:\n")
                    f.write("\n".join(stderr_lines) + "\n")
            
            print("="*80)
            print(f"[{stage_name}] 结束时间: {datetime.now().isoformat()}")
            print(f"[{stage_name}] 执行时间: {end_time - start_time:.2f}秒")
            print(f"[{stage_name}] 返回码: {return_code}")
            print(f"[{stage_name}] 标准输出行数: {len(stdout_lines)}")
            print(f"[{stage_name}] 标准错误行数: {len(stderr_lines)}")
            print("="*80)
            
            result = {
                'success': return_code == 0,
                'total_duration': end_time - start_time,
                'subprocess_duration': end_time - start_time,
                'return_code': return_code,
            }
            
            if return_code != 0:
                result['error'] = '\n'.join(stderr_lines) if stderr_lines else 'Unknown error'
                print(f"[{stage_name}] 执行失败")
            else:
                print(f"[{stage_name}] 执行成功")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            
            # 写入错误日志
            with open(log_file, 'w') as f:
                f.write(f"执行命令: {' '.join(cmd)}\n")
                f.write(f"开始时间: {datetime.now().isoformat()}\n")
                f.write("="*80 + "\n")
                f.write(f"执行异常: {e}\n")
                f.write(f"结束时间: {datetime.now().isoformat()}\n")
                f.write(f"执行时间: {end_time - start_time:.2f}秒\n")
            
            print("="*80)
            print(f"[{stage_name}] 执行异常: {e}")
            print("="*80)
            
            return {
                'success': False,
                'total_duration': end_time - start_time,
                'subprocess_duration': end_time - start_time,
                'return_code': -2,
                'error': str(e),
            }
    
    def run_sensitivity_analysis(self, parameter_ranges: Dict[str, List], 
                                max_workers: int = 1) -> str:
        """
        运行参数敏感性分析
        
        Args:
            parameter_ranges: 参数范围字典，如 {'gen_num': [2, 4, 8], 'batch_size': [16, 32, 64]}
            max_workers: 最大并行工作线程数
        
        Returns:
            分析报告文件路径
        """
        print(f"\n=== 开始参数敏感性分析 ===")
        print(f"参数范围: {parameter_ranges}")
        print(f"最大并行数: {max_workers}")
        print("超时设置: 无限制（实时输出）")
        
        # 生成实验配置列表
        experiment_configs = self._generate_experiment_configs(parameter_ranges)
        print(f"总实验数量: {len(experiment_configs)}")
        
        # 执行实验
        if max_workers == 1:
            # 串行执行
            for i, config in enumerate(experiment_configs):
                print(f"\n--- 执行实验 {i+1}/{len(experiment_configs)} ---")
                result = self._run_single_experiment(config)
                self.results.append(result)
        else:
            # 并行执行
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {
                    executor.submit(self._run_single_experiment, config): config 
                    for config in experiment_configs
                }
                
                for i, future in enumerate(as_completed(future_to_config)):
                    config = future_to_config[future]
                    print(f"\n--- 完成实验 {i+1}/{len(experiment_configs)} ---")
                    try:
                        result = future.result()
                        with self.lock:
                            self.results.append(result)
                    except Exception as e:
                        print(f"实验失败: {config}, 错误: {e}")
                        with self.lock:
                            self.results.append({
                                'config': config,
                                'success': False,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            })
        
        # 生成分析报告
        report_file = self._generate_analysis_report()
        
        print(f"\n=== 参数敏感性分析完成 ===")
        print(f"总实验数: {len(experiment_configs)}")
        print(f"成功实验数: {len([r for r in self.results if r.get('success', False)])}")
        print(f"分析报告: {report_file}")
        
        return report_file
    
    def _generate_experiment_configs(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """生成实验配置列表"""
        configs = []
        
        for param_name, param_values in parameter_ranges.items():
            for param_value in param_values:
                # 创建基于base_config的配置副本
                config = self.base_config.copy()
                config[param_name] = param_value
                config['variable_param'] = param_name
                config['variable_value'] = param_value
                config['exp_id'] = f"{param_name}_{param_value}_{datetime.now().strftime('%H%M%S')}"
                configs.append(config)
        
        return configs
    
    def _run_single_experiment(self, config: Dict) -> Dict:
        """运行单个实验"""
        exp_id = config['exp_id']
        param_name = config['variable_param']
        param_value = config['variable_value']
        
        print(f"开始实验: {exp_id} ({param_name}={param_value})")
        
        # 创建性能监控器
        monitor = PerformanceMonitor(
            exp_id, 
            os.path.join(self.output_dir, "performance_data")
        )
        
        experiment_metadata = {
            'experiment_type': 'parameter_sensitivity',
            'variable_parameter': param_name,
            'variable_value': param_value,
            'config': config
        }
        
        start_time = time.time()
        
        with monitor.monitor_stage("full_experiment", experiment_metadata):
            # 运行简化的TeR-TSF流程（仅数据准备阶段，避免完整训练）
            try:
                result = self._run_data_preparation_experiment(config)
                
                end_time = time.time()
                total_duration = end_time - start_time
                
                # 保存监控结果
                stage_file, timeline_file, summary_file = monitor.save_results()
                
                # 提取性能指标
                performance_metrics = self._extract_performance_metrics(
                    stage_file, timeline_file, summary_file
                )
                
                experiment_result = {
                    'exp_id': exp_id,
                    'variable_param': param_name,
                    'variable_value': param_value,
                    'config': config,
                    'success': result['success'],
                    'total_duration': total_duration,
                    'subprocess_duration': result.get('duration', 0),
                    'return_code': result.get('return_code', 0),
                    'performance_metrics': performance_metrics,
                    'stage_file': stage_file,
                    'timeline_file': timeline_file,
                    'summary_file': summary_file,
                    'timestamp': datetime.now().isoformat(),
                }
                
                if not result['success']:
                    experiment_result['error'] = result.get('error', 'Unknown error')
                
                print(f"实验完成: {exp_id}, 成功: {result['success']}, 耗时: {total_duration:.2f}秒")
                
            except Exception as e:
                end_time = time.time()
                total_duration = end_time - start_time
                
                experiment_result = {
                    'exp_id': exp_id,
                    'variable_param': param_name,
                    'variable_value': param_value,
                    'config': config,
                    'success': False,
                    'total_duration': total_duration,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                }
                
                print(f"实验异常: {exp_id}, 错误: {e}")
        
        return experiment_result
    
    def _run_data_preparation_experiment(self, config: Dict) -> Dict:
        """运行数据准备实验（简化版）"""
        # 构建prepare_stage.py命令
        cmd = [
            'python', 'prepare_stage.py',
            '--data_dir', '/data2/user2/ter_tsf',
            '--data_name', config['data_name'],
            '--llm_type', config['llm_type'],
            '--tsf_type', config['tsf_type'],
            '--hist_len', str(config['hist_len']),
            '--pred_len', str(config['pred_len']),
            '--batch_size', str(config['batch_size']),
            '--gen_num', str(config['gen_num']),
            '--iter_idx', '0',  # 只运行第0轮
            '--llm_path', 'original',
            '--exp_time', config['exp_id'],
            '--llama_factory_dir', config.get('llama_factory_dir', 'llama-factory-main'),
            '--down_sample', str(config.get('down_sample', 0)),
            '--max_batches', '1',  # 仅处理1个batch以节省时间
        ]
        
        # 添加可选参数
        if config.get('disable_text_quality_reward', False):
            cmd.append('--disable_text_quality_reward')
        
        # 执行命令（无超时限制，实时输出）
        log_file = os.path.join(self.output_dir, "logs", f"{config['exp_id']}.log")
        
        # 在日志文件开头写入实验配置
        with open(log_file, 'w') as f:
            f.write(f"实验配置: {json.dumps(config, indent=2)}\n")
        
        result = self._run_command_with_realtime_output(cmd, config['exp_id'], log_file)
        
        return {
            'success': result['success'],
            'duration': result['total_duration'],
            'return_code': result['return_code'],
            'stdout_lines': 0,  # 已在实时输出中处理
            'stderr_lines': 0,  # 已在实时输出中处理
            'command': ' '.join(cmd),
            'log_file': log_file,
        }
    
    def _extract_performance_metrics(self, stage_file: str, timeline_file: Optional[str], 
                                   summary_file: str) -> Dict:
        """提取性能指标"""
        metrics = {}
        
        try:
            # 读取阶段统计数据
            if os.path.exists(stage_file):
                stage_df = pd.read_csv(stage_file)
                if not stage_df.empty:
                    metrics.update({
                        'total_duration_seconds': stage_df['duration_seconds'].sum(),
                        'peak_memory_gb': stage_df['memory_peak_used_gb'].max(),
                        'avg_cpu_usage_percent': stage_df['cpu_usage_avg_percent'].mean(),
                        'total_memory_delta_gb': stage_df['memory_delta_gb'].sum(),
                        'stage_count': len(stage_df),
                    })
                    
                    # GPU指标
                    if 'gpu_memory_peak_mb' in stage_df.columns:
                        metrics.update({
                            'peak_gpu_memory_mb': stage_df['gpu_memory_peak_mb'].max(),
                            'total_gpu_memory_delta_mb': stage_df['gpu_memory_delta_mb'].sum(),
                        })
            
            # 读取汇总报告
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    
                metrics.update({
                    'experiment_duration_minutes': summary_data.get('experiment_info', {}).get('total_duration_minutes', 0),
                    'bottleneck_stage': summary_data.get('bottleneck_analysis', {}).get('slowest_stage', ''),
                    'memory_intensive_stage': summary_data.get('bottleneck_analysis', {}).get('most_memory_intensive_stage', ''),
                })
            
            # 计算效率指标
            if metrics.get('total_duration_seconds', 0) > 0:
                metrics['throughput'] = 1.0 / metrics['total_duration_seconds']  # 实验/秒
                metrics['memory_efficiency'] = metrics.get('throughput', 0) / max(metrics.get('peak_memory_gb', 1), 0.1)
                
                if metrics.get('peak_gpu_memory_mb', 0) > 0:
                    metrics['gpu_efficiency'] = metrics.get('throughput', 0) / (metrics.get('peak_gpu_memory_mb', 1) / 1024)  # 考虑GPU内存使用
            
        except Exception as e:
            print(f"性能指标提取失败: {e}")
            metrics['extraction_error'] = str(e)
        
        return metrics
    
    def _generate_analysis_report(self) -> str:
        """生成分析报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始结果
        results_file = os.path.join(self.output_dir, f"raw_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 创建结构化分析
        analysis_data = []
        for result in self.results:
            if result.get('success', False):
                row = {
                    'exp_id': result['exp_id'],
                    'variable_param': result['variable_param'],
                    'variable_value': result['variable_value'],
                    'total_duration': result['total_duration'],
                    'subprocess_duration': result.get('subprocess_duration', 0),
                    'success': result['success'],
                }
                
                # 添加性能指标
                perf_metrics = result.get('performance_metrics', {})
                for key, value in perf_metrics.items():
                    if isinstance(value, (int, float)):
                        row[f'perf_{key}'] = value
                
                # 添加配置参数
                config = result.get('config', {})
                for param in ['data_name', 'llm_type', 'tsf_type', 'hist_len', 'pred_len', 'batch_size', 'gen_num']:
                    if param in config:
                        row[f'config_{param}'] = config[param]
                
                analysis_data.append(row)
        
        # 创建DataFrame并保存
        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            csv_file = os.path.join(self.output_dir, f"analysis_results_{timestamp}.csv")
            analysis_df.to_csv(csv_file, index=False)
            
            # 生成统计分析
            stats_report = self._generate_statistical_analysis(analysis_df)
        else:
            csv_file = None
            stats_report = {'error': 'No successful experiments to analyze'}
        
        # 生成综合报告
        comprehensive_report = {
            'experiment_info': {
                'timestamp': timestamp,
                'total_experiments': len(self.results),
                'successful_experiments': len([r for r in self.results if r.get('success', False)]),
                'failed_experiments': len([r for r in self.results if not r.get('success', False)]),
                'base_config': self.base_config,
            },
            'files': {
                'raw_results': results_file,
                'analysis_csv': csv_file,
            },
            'statistical_analysis': stats_report,
            'parameter_impact_summary': self._generate_parameter_impact_summary(),
        }
        
        # 保存综合报告
        report_file = os.path.join(self.output_dir, f"comprehensive_report_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存:")
        print(f"  - 原始结果: {results_file}")
        if csv_file:
            print(f"  - 结构化数据: {csv_file}")
        print(f"  - 综合报告: {report_file}")
        
        return report_file
    
    def _generate_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """生成统计分析"""
        stats = {}
        
        try:
            # 按参数类型分组分析
            for param_name in df['variable_param'].unique():
                param_data = df[df['variable_param'] == param_name].copy()
                param_data = param_data.sort_values('variable_value')
                
                if len(param_data) > 1:
                    # 计算相关性
                    correlation_metrics = {}
                    for metric in ['total_duration', 'perf_peak_memory_gb', 'perf_throughput']:
                        if metric in param_data.columns:
                            corr = param_data[['variable_value', metric]].corr().iloc[0, 1]
                            correlation_metrics[metric] = corr if not np.isnan(corr) else 0
                    
                    stats[param_name] = {
                        'sample_count': len(param_data),
                        'value_range': [param_data['variable_value'].min(), param_data['variable_value'].max()],
                        'duration_stats': {
                            'mean': param_data['total_duration'].mean(),
                            'std': param_data['total_duration'].std(),
                            'min': param_data['total_duration'].min(),
                            'max': param_data['total_duration'].max(),
                        },
                        'correlations': correlation_metrics,
                        'best_config': {
                            'value': param_data.loc[param_data['perf_memory_efficiency'].idxmax(), 'variable_value'] if 'perf_memory_efficiency' in param_data.columns else param_data.loc[param_data['total_duration'].idxmin(), 'variable_value'],
                            'metric': 'memory_efficiency' if 'perf_memory_efficiency' in param_data.columns else 'duration',
                        }
                    }
                    
                    # 内存使用统计
                    if 'perf_peak_memory_gb' in param_data.columns:
                        stats[param_name]['memory_stats'] = {
                            'mean': param_data['perf_peak_memory_gb'].mean(),
                            'std': param_data['perf_peak_memory_gb'].std(),
                            'min': param_data['perf_peak_memory_gb'].min(),
                            'max': param_data['perf_peak_memory_gb'].max(),
                        }
        
        except Exception as e:
            stats['error'] = f"Statistical analysis failed: {e}"
        
        return stats
    
    def _generate_parameter_impact_summary(self) -> Dict:
        """生成参数影响摘要"""
        summary = {
            'parameters_tested': [],
            'impact_ranking': [],
            'recommendations': [],
        }
        
        try:
            # 统计测试的参数
            for result in self.results:
                if result.get('success', False):
                    param_name = result.get('variable_param')
                    if param_name and param_name not in summary['parameters_tested']:
                        summary['parameters_tested'].append(param_name)
            
            # 简单的影响排序（基于持续时间变化）
            param_impact = {}
            for param_name in summary['parameters_tested']:
                param_results = [r for r in self.results if r.get('variable_param') == param_name and r.get('success', False)]
                if len(param_results) > 1:
                    durations = [r['total_duration'] for r in param_results]
                    impact_score = (max(durations) - min(durations)) / min(durations) if min(durations) > 0 else 0
                    param_impact[param_name] = impact_score
            
            # 按影响程度排序
            summary['impact_ranking'] = sorted(param_impact.items(), key=lambda x: x[1], reverse=True)
            
            # 生成建议
            if summary['impact_ranking']:
                highest_impact_param = summary['impact_ranking'][0][0]
                summary['recommendations'].append(f"参数 {highest_impact_param} 对性能影响最大，建议重点优化")
                
                if len(summary['impact_ranking']) > 1:
                    lowest_impact_param = summary['impact_ranking'][-1][0]
                    summary['recommendations'].append(f"参数 {lowest_impact_param} 对性能影响较小，可以优先考虑其他因素")
        
        except Exception as e:
            summary['error'] = f"Impact summary generation failed: {e}"
        
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TeR-TSF 参数敏感性分析")
    
    # 基础配置参数
    parser.add_argument("--data_name", type=str, default="Energy", help="数据集名称")
    parser.add_argument("--llm_type", type=str, default="qwen3-1.7b", help="LLM类型")
    parser.add_argument("--tsf_type", type=str, default="tfhts", help="TSF类型")
    parser.add_argument("--hist_len", type=int, default=96, help="历史序列长度")
    parser.add_argument("--pred_len", type=int, default=12, help="预测序列长度")
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小")
    parser.add_argument("--gen_num", type=int, default=2, help="生成文本数量")
    parser.add_argument("--llama_factory_dir", type=str, default="llama-factory-main", help="LLaMA Factory目录")
    parser.add_argument("--down_sample", type=int, default=0, help="下采样")
    parser.add_argument("--disable_text_quality_reward", action="store_true", help="禁用文本质量奖励")
    
    # 实验配置参数
    parser.add_argument("--output_dir", type=str, default="./param_sensitivity_results", help="输出目录")
    parser.add_argument("--max_workers", type=int, default=1, help="最大并行工作线程数")
    # 移除超时参数，使用实时输出模式
    
    # 参数范围配置
    parser.add_argument("--test_gen_num", nargs='+', type=int, default=[2, 4, 8], help="测试的gen_num值")
    parser.add_argument("--test_batch_size", nargs='+', type=int, default=[16, 32, 64], help="测试的batch_size值")
    parser.add_argument("--test_hist_len", nargs='+', type=int, default=[36, 96, 192], help="测试的hist_len值")
    parser.add_argument("--test_pred_len", nargs='+', type=int, default=[6, 12, 24], help="测试的pred_len值")
    
    args = parser.parse_args()
    
    # 构建基础配置
    base_config = {
        'data_name': args.data_name,
        'llm_type': args.llm_type,
        'tsf_type': args.tsf_type,
        'hist_len': args.hist_len,
        'pred_len': args.pred_len,
        'batch_size': args.batch_size,
        'gen_num': args.gen_num,
        'llama_factory_dir': args.llama_factory_dir,
        'down_sample': args.down_sample,
        'disable_text_quality_reward': args.disable_text_quality_reward,
    }
    
    # 构建参数范围
    parameter_ranges = {}
    if len(args.test_gen_num) > 1:
        parameter_ranges['gen_num'] = args.test_gen_num
    if len(args.test_batch_size) > 1:
        parameter_ranges['batch_size'] = args.test_batch_size
    if len(args.test_hist_len) > 1:
        parameter_ranges['hist_len'] = args.test_hist_len
    if len(args.test_pred_len) > 1:
        parameter_ranges['pred_len'] = args.test_pred_len
    
    if not parameter_ranges:
        print("错误：没有指定要测试的参数范围")
        print("请使用 --test_* 参数指定至少两个不同的值")
        sys.exit(1)
    
    print(f"将测试的参数范围: {parameter_ranges}")
    
    # 创建分析器并运行分析
    analyzer = ParameterSensitivityAnalyzer(base_config, args.output_dir)
    report_file = analyzer.run_sensitivity_analysis(
        parameter_ranges, 
        max_workers=args.max_workers, 
        # 不再需要timeout参数
    )
    
    print(f"\n=== 参数敏感性分析完成 ===")
    print(f"详细报告: {report_file}")
    print("可以使用生成的CSV文件进行进一步的数据分析和可视化")


if __name__ == "__main__":
    main() 