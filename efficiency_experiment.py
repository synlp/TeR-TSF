#!/usr/bin/env python3
"""
TeR-TSF 模型效率测试实验
测量不同数据集和参数配置下的运行时间

实验设计：
- 数据集：可配置（默认Energy）
- 预测长度：可配置（默认[12, 24, 48]）
- 批处理：可配置batch_size（默认64），仅处理1个batch
- 迭代设置：iteration=1，epoch=1，dpo_epoch=1
- 测量阶段：数据准备阶段（文本生成）+ DPO阶段
- 排除阶段：evaluation阶段
"""

import subprocess
import os
import sys
import json
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import threading
from performance_monitor import PerformanceMonitor


class EfficiencyExperimentRunner:
    """效率测试实验运行器"""
    
    def __init__(self, base_config: Dict, experiment_config: Dict, output_dir: str = "./efficiency_results"):
        self.base_config = base_config
        self.experiment_config = experiment_config
        self.output_dir = output_dir
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "performance_data"), exist_ok=True)
        
        # 保存实验配置
        self.config_file = os.path.join(output_dir, "efficiency_experiment_config.json")
        with open(self.config_file, 'w') as f:
            json.dump({
                'experiment_type': 'efficiency_experiment',
                'timestamp': datetime.now().isoformat(),
                'base_config': base_config,
                'experiment_config': experiment_config,
                'output_dir': output_dir,
                'test_pred_lengths': experiment_config['pred_len'],
                'fixed_settings': {
                    'max_batches': 1,
                    'iteration': 1,
                    'epoch': 1,
                    'dpo_epoch': 1
                }
            }, f, indent=2)
        
        print(f"效率测试实验已初始化")
        print(f"基础配置: {base_config}")
        print(f"实验配置: {experiment_config}")
        print(f"输出目录: {output_dir}")
        print(f"数据集: {experiment_config['data_name']}")
        print(f"测试预测长度: {experiment_config['pred_len']}")
        print(f"批处理大小: {experiment_config['batch_size']}")
    
    def _run_command_with_realtime_output(self, cmd: List[str], stage_name: str, log_file: str) -> Dict:
        """运行命令并实时显示输出"""
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
                'duration': end_time - start_time,
                'return_code': return_code,
                'stdout_lines': len(stdout_lines),
                'stderr_lines': len(stderr_lines),
                'command': ' '.join(cmd),
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
                'duration': end_time - start_time,
                'return_code': -2,
                'error': str(e),
                'command': ' '.join(cmd),
            }
    
    def run_efficiency_experiment(self) -> str:
        """运行效率测试实验"""
        print(f"\n=== 开始效率测试实验 ===")
        print(f"数据集: {self.experiment_config['data_name']}")
        print(f"测试预测长度: {self.experiment_config['pred_len']}")
        print(f"批处理设置: batch_size={self.experiment_config['batch_size']}, max_batches=1")
        print(f"迭代设置: iteration=1, epoch=1, dpo_epoch=1")
        print(f"测量阶段: 数据准备阶段 + DPO阶段")
        
        pred_lengths = self.experiment_config['pred_len']
        
        for pred_len in pred_lengths:
            print(f"\n--- 测试预测长度: {pred_len} ---")
            result = self._run_single_pred_len_experiment(pred_len)
            self.results.append(result)
        
        # 生成分析报告
        report_file = self._generate_efficiency_report()
        
        print(f"\n=== 效率测试实验完成 ===")
        print(f"总测试数: {len(pred_lengths)}")
        print(f"成功测试数: {len([r for r in self.results if r.get('overall_success', False)])}")
        print(f"效率报告: {report_file}")
        
        return report_file
    
    def _run_single_pred_len_experiment(self, pred_len: int) -> Dict:
        """运行单个预测长度的效率测试"""
        exp_id = f"efficiency_Energy_pred{pred_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"开始实验: {exp_id}")
        
        # 创建性能监控器
        monitor = PerformanceMonitor(exp_id, os.path.join(self.output_dir, "performance_data"))
        
        experiment_metadata = {
            'experiment_type': 'efficiency_test',
            'data_name': self.experiment_config['data_name'],
            'pred_len': pred_len,
            'hist_len': self.base_config['hist_len'],
            'batch_size': self.experiment_config['batch_size'],
            'max_batches': 1,
            'llm_type': self.base_config['llm_type'],
            'tsf_type': self.base_config['tsf_type'],
        }
        
        start_time = time.time()
        
        with monitor.monitor_stage("full_efficiency_experiment", experiment_metadata):
            try:
                # 阶段1: 数据准备阶段（文本生成）
                data_prep_result = self._run_data_preparation_stage(exp_id, pred_len, monitor)
                
                # 阶段2: DPO训练阶段
                dpo_result = self._run_dpo_training_stage(exp_id, pred_len, monitor)
                
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
                    'pred_len': pred_len,
                    'data_name': self.experiment_config['data_name'],
                    'total_duration': total_duration,
                    'data_prep_result': data_prep_result,
                    'dpo_result': dpo_result,
                    'overall_success': data_prep_result['success'] and dpo_result['success'],
                    'performance_metrics': performance_metrics,
                    'stage_file': stage_file,
                    'timeline_file': timeline_file,
                    'summary_file': summary_file,
                    'timestamp': datetime.now().isoformat(),
                }
                
                print(f"实验完成: {exp_id}, 成功: {experiment_result['overall_success']}, 总耗时: {total_duration:.2f}秒")
                
            except Exception as e:
                end_time = time.time()
                total_duration = end_time - start_time
                
                experiment_result = {
                    'exp_id': exp_id,
                    'pred_len': pred_len,
                    'data_name': self.experiment_config['data_name'],
                    'total_duration': total_duration,
                    'overall_success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                }
                
                print(f"实验异常: {exp_id}, 错误: {e}")
        
        return experiment_result
    
    def _run_data_preparation_stage(self, exp_id: str, pred_len: int, monitor: PerformanceMonitor) -> Dict:
        """运行数据准备阶段"""
        print(f"\n--- 数据准备阶段 (pred_len={pred_len}) ---")
        
        stage_metadata = {
            'stage_type': 'data_preparation',
            'pred_len': pred_len,
            'batch_size': self.experiment_config['batch_size'],
            'max_batches': 1,
            'gen_num': self.base_config['gen_num'],
        }
        
        with monitor.monitor_stage("data_preparation", stage_metadata):
            # 构建prepare_stage.py命令
            cmd = [
                'python', 'prepare_stage.py',
                '--data_dir', '/data2/user2/ter_tsf',
                '--data_name', self.experiment_config['data_name'],
                '--llm_type', self.base_config['llm_type'],
                '--tsf_type', self.base_config['tsf_type'],
                '--hist_len', str(self.base_config['hist_len']),
                '--pred_len', str(pred_len),
                '--batch_size', str(self.experiment_config['batch_size']),
                '--gen_num', str(self.base_config['gen_num']),
                '--iter_idx', '0',  # 固定为第0轮
                '--llm_path', 'original',
                '--exp_time', f"efficiency_{exp_id}",
                '--llama_factory_dir', self.base_config.get('llama_factory_dir', 'llama-factory-main'),
                '--down_sample', str(self.base_config.get('down_sample', 0)),
                '--max_batches', '1',  # 仅处理1个batch
            ]
            
            # 如果禁用文本质量奖励
            if self.base_config.get('disable_text_quality_reward', False):
                cmd.append('--disable_text_quality_reward')
            
            # 执行命令
            log_file = os.path.join(self.output_dir, "logs", f"{exp_id}_data_prep.log")
            stage_result = self._run_command_with_realtime_output(cmd, f"数据准备_pred{pred_len}", log_file)
            stage_result['stage_name'] = 'data_preparation'
            stage_result['pred_len'] = pred_len
        
        return stage_result
    
    def _run_dpo_training_stage(self, exp_id: str, pred_len: int, monitor: PerformanceMonitor) -> Dict:
        """运行DPO训练阶段"""
        print(f"\n--- DPO训练阶段 (pred_len={pred_len}) ---")
        
        stage_metadata = {
            'stage_type': 'dpo_training',
            'pred_len': pred_len,
            'llm_type': self.base_config['llm_type'],
            'num_train_epochs': 1,  # 固定为1个epoch
            'per_device_train_batch_size': self.base_config.get('per_device_train_batch_size', 2),
            'lora_rank': self.base_config.get('lora_rank', 8),
        }
        
        with monitor.monitor_stage("dpo_training", stage_metadata):
            try:
                # 检查偏好数据是否存在
                preference_data_path = f"/data2/user2/ter_tsf/{self.base_config['llm_type']}/{self.base_config['tsf_type']}/{self.experiment_config['data_name']}/preference_data"
                
                if not os.path.exists(preference_data_path):
                    print(f"偏好数据路径不存在: {preference_data_path}")
                    return {
                        'stage_name': 'dpo_training',
                        'pred_len': pred_len,
                        'success': False,
                        'duration': 0,
                        'return_code': -3,
                        'error': 'Preference data not found',
                        'skipped': True,
                    }
                
                # 构建DPO训练命令
                llm_config = self._get_llm_config(self.base_config['llm_type'])
                llm_template, initial_path = llm_config.split(':')
                
                dataset_name = f"{self.experiment_config['data_name']}_h{self.base_config['hist_len']}_p{pred_len}_{self.base_config['llm_type']}_{self.base_config['tsf_type']}_genNum{self.base_config['gen_num']}_iter0_efficiency_{exp_id}"
                
                adapter_output_dir = f"/data2/user2/ter_tsf/{self.base_config['llm_type']}/dpo/{self.experiment_config['data_name']}/iter0_efficiency_{exp_id}"
                
                # 切换到llama-factory目录
                llama_factory_dir = self.base_config.get('llama_factory_dir', 'llama-factory-main')
                
                cmd = [
                    'bash', '-c', f'''
                    cd {llama_factory_dir} && \\
                    llamafactory-cli train \\
                        --model_name_or_path {initial_path} \\
                        --stage dpo \\
                        --do_train \\
                        --finetuning_type lora \\
                        --lora_rank {self.base_config.get('lora_rank', 8)} \\
                        --lora_target all \\
                        --pref_beta 0.1 \\
                        --pref_loss sigmoid \\
                        --dataset {dataset_name} \\
                        --dataset_dir "/home/user2/projects/TeR_TSF/llama-factory-main/data" \\
                        --template {llm_template} \\
                        --cutoff_len 5120 \\
                        --max_samples 100 \\
                        --overwrite_cache \\
                        --preprocessing_num_workers 16 \\
                        --dataloader_num_workers 16 \\
                        --output_dir {adapter_output_dir} \\
                        --logging_steps 5 \\
                        --save_steps 50 \\
                        --per_device_train_batch_size {self.base_config.get('per_device_train_batch_size', 2)} \\
                        --gradient_accumulation_steps 4 \\
                        --learning_rate {self.base_config.get('lr', 5.0e-5)} \\
                        --num_train_epochs 1 \\
                        --lr_scheduler_type cosine \\
                        --warmup_ratio 0.1 \\
                        --bf16 \\
                        --ddp_timeout 180000000
                    '''
                ]
                
                # 执行命令
                log_file = os.path.join(self.output_dir, "logs", f"{exp_id}_dpo_pred{pred_len}.log")
                stage_result = self._run_command_with_realtime_output(cmd, f"DPO训练_pred{pred_len}", log_file)
                stage_result['stage_name'] = 'dpo_training'
                stage_result['pred_len'] = pred_len
                stage_result['dataset_name'] = dataset_name
                stage_result['adapter_output_dir'] = adapter_output_dir
                
            except Exception as e:
                stage_result = {
                    'stage_name': 'dpo_training',
                    'pred_len': pred_len,
                    'success': False,
                    'duration': 0,
                    'return_code': -2,
                    'error': str(e),
                }
                print(f"DPO训练阶段异常: {e}")
        
        return stage_result
    
    def _get_llm_config(self, llm_type: str) -> str:
        """获取LLM配置（与main.sh保持一致）"""
        llm_configs = {
            'qwen3-1.7b': 'qwen3:/data2/user2/Qwen3-1.7B',
            'qwen3-8b': 'qwen3:/data2/user2/Qwen3-8B',
            'qwen3-4b': 'qwen3:/data2/user2/Qwen3-4B',
            'llama3.1-8b': 'llama3:/data2/user2/Llama-3.1-8B',
            'llama3.2-1b': 'llama3:/data2/user2/Llama-3.2-1B',
            'llama3.2-3b': 'llama3:/data2/user2/Llama-3.2-3B',
        }
        
        return llm_configs.get(llm_type, 'qwen3:/data2/user2/Qwen3-1.7B')
    
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
                    
                    # 按阶段分解
                    for stage_name in stage_df['stage_name'].unique():
                        stage_data = stage_df[stage_df['stage_name'] == stage_name]
                        metrics[f'{stage_name}_duration'] = stage_data['duration_seconds'].sum()
                        metrics[f'{stage_name}_peak_memory'] = stage_data['memory_peak_used_gb'].max()
                        metrics[f'{stage_name}_avg_cpu'] = stage_data['cpu_usage_avg_percent'].mean()
                    
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
                })
        
        except Exception as e:
            print(f"性能指标提取失败: {e}")
            metrics['extraction_error'] = str(e)
        
        return metrics
    
    def _generate_efficiency_report(self) -> str:
        """生成效率测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始结果
        results_file = os.path.join(self.output_dir, f"efficiency_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 创建结构化分析
        analysis_data = []
        for result in self.results:
            if result.get('overall_success', False):
                row = {
                    'pred_len': result['pred_len'],
                    'total_duration': result['total_duration'],
                    'data_prep_duration': result.get('data_prep_result', {}).get('duration', 0),
                    'dpo_duration': result.get('dpo_result', {}).get('duration', 0),
                    'data_prep_success': result.get('data_prep_result', {}).get('success', False),
                    'dpo_success': result.get('dpo_result', {}).get('success', False),
                    'overall_success': result['overall_success'],
                }
                
                # 添加性能指标
                perf_metrics = result.get('performance_metrics', {})
                for key, value in perf_metrics.items():
                    if isinstance(value, (int, float)):
                        row[f'perf_{key}'] = value
                
                analysis_data.append(row)
        
        # 创建DataFrame并保存
        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            csv_file = os.path.join(self.output_dir, f"efficiency_analysis_{timestamp}.csv")
            analysis_df.to_csv(csv_file, index=False)
            
            # 生成效率分析
            efficiency_stats = self._generate_efficiency_analysis(analysis_df)
        else:
            csv_file = None
            efficiency_stats = {'error': 'No successful experiments to analyze'}
        
        # 生成综合报告
        comprehensive_report = {
            'experiment_info': {
                'timestamp': timestamp,
                'experiment_type': 'efficiency_experiment',
                'data_name': self.experiment_config['data_name'],
                'test_pred_lengths': self.experiment_config['pred_len'],
                'batch_size': self.experiment_config['batch_size'],
                'total_experiments': len(self.results),
                'successful_experiments': len([r for r in self.results if r.get('overall_success', False)]),
                'base_config': self.base_config,
                'experiment_config': self.experiment_config,
            },
            'files': {
                'raw_results': results_file,
                'analysis_csv': csv_file,
            },
            'efficiency_analysis': efficiency_stats,
            'summary': {
                'fastest_pred_len': None,
                'slowest_pred_len': None,
                'efficiency_trend': 'N/A',
            }
        }
        
        # 添加效率趋势分析
        if analysis_data:
            sorted_results = sorted(analysis_data, key=lambda x: x['total_duration'])
            comprehensive_report['summary'].update({
                'fastest_pred_len': sorted_results[0]['pred_len'],
                'slowest_pred_len': sorted_results[-1]['pred_len'],
                'efficiency_trend': 'Duration increases with prediction length' if sorted_results[0]['pred_len'] < sorted_results[-1]['pred_len'] else 'Mixed trend',
            })
        
        # 保存综合报告
        report_file = os.path.join(self.output_dir, f"efficiency_comprehensive_report_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        print(f"效率分析结果已保存:")
        print(f"  - 原始结果: {results_file}")
        if csv_file:
            print(f"  - 结构化数据: {csv_file}")
        print(f"  - 综合报告: {report_file}")
        
        return report_file
    
    def _generate_efficiency_analysis(self, df: pd.DataFrame) -> Dict:
        """生成效率分析"""
        stats = {}
        
        try:
            # 按预测长度分析
            stats['pred_len_analysis'] = {}
            for pred_len in df['pred_len'].unique():
                pred_data = df[df['pred_len'] == pred_len].iloc[0]
                stats['pred_len_analysis'][f'pred_{pred_len}'] = {
                    'total_duration': pred_data['total_duration'],
                    'data_prep_duration': pred_data['data_prep_duration'],
                    'dpo_duration': pred_data['dpo_duration'],
                    'data_prep_ratio': pred_data['data_prep_duration'] / pred_data['total_duration'] if pred_data['total_duration'] > 0 else 0,
                    'dpo_ratio': pred_data['dpo_duration'] / pred_data['total_duration'] if pred_data['total_duration'] > 0 else 0,
                }
            
            # 整体统计
            stats['overall_stats'] = {
                'total_duration_range': [df['total_duration'].min(), df['total_duration'].max()],
                'avg_total_duration': df['total_duration'].mean(),
                'avg_data_prep_duration': df['data_prep_duration'].mean(),
                'avg_dpo_duration': df['dpo_duration'].mean(),
                'data_prep_avg_ratio': (df['data_prep_duration'] / df['total_duration']).mean(),
                'dpo_avg_ratio': (df['dpo_duration'] / df['total_duration']).mean(),
            }
            
            # 效率排名
            sorted_by_total = df.sort_values('total_duration')
            stats['efficiency_ranking'] = {
                'fastest': {
                    'pred_len': int(sorted_by_total.iloc[0]['pred_len']),
                    'duration': sorted_by_total.iloc[0]['total_duration'],
                },
                'slowest': {
                    'pred_len': int(sorted_by_total.iloc[-1]['pred_len']),
                    'duration': sorted_by_total.iloc[-1]['total_duration'],
                }
            }
            
        except Exception as e:
            stats['error'] = f"Efficiency analysis failed: {e}"
        
        return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TeR-TSF 效率测试实验")
    
    # 基础配置参数
    parser.add_argument("--llm_type", type=str, default="qwen3-1.7b", help="LLM类型")
    parser.add_argument("--tsf_type", type=str, default="tfhts", help="TSF类型")
    parser.add_argument("--hist_len", type=int, default=336, help="历史序列长度")
    parser.add_argument("--gen_num", type=int, default=2, help="生成文本数量")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每设备训练批次大小")
    parser.add_argument("--llama_factory_dir", type=str, default="llama-factory-main", help="LLaMA Factory目录")
    parser.add_argument("--down_sample", type=int, default=0, help="下采样")
    parser.add_argument("--disable_text_quality_reward", action="store_true", help="禁用文本质量奖励")
    
    # 实验配置参数
    parser.add_argument("--data_name", type=str, default="ETTh1", help="数据集名称")
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小")
    parser.add_argument("--pred_len", nargs='+', type=int, default=[96, 192, 336], help="预测长度列表")
    parser.add_argument("--output_dir", type=str, default="./efficiency_results", help="输出目录")
    
    args = parser.parse_args()
    
    # 构建基础配置
    base_config = {
        'llm_type': args.llm_type,
        'tsf_type': args.tsf_type,
        'hist_len': args.hist_len,
        'gen_num': args.gen_num,
        'lora_rank': args.lora_rank,
        'lr': args.lr,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'llama_factory_dir': args.llama_factory_dir,
        'down_sample': args.down_sample,
        'disable_text_quality_reward': args.disable_text_quality_reward,
    }
    
    # 构建实验配置
    experiment_config = {
        'data_name': args.data_name,
        'batch_size': args.batch_size,
        'pred_len': args.pred_len,
    }
    
    print(f"效率测试实验配置:")
    print(f"  - 数据集: {args.data_name}")
    print(f"  - 预测长度: {args.pred_len}")
    print(f"  - LLM类型: {args.llm_type}")
    print(f"  - TSF类型: {args.tsf_type}")
    print(f"  - 历史长度: {args.hist_len}")
    print(f"  - 批处理: batch_size={args.batch_size}, max_batches=1")
    print(f"  - 迭代设置: iteration=1, epoch=1, dpo_epoch=1")
    
    # 创建实验运行器并执行
    runner = EfficiencyExperimentRunner(base_config, experiment_config, args.output_dir)
    report_file = runner.run_efficiency_experiment()
    
    print(f"\n=== 效率测试实验完成 ===")
    print(f"详细报告: {report_file}")
    print("可以使用生成的CSV文件进行进一步的效率分析和可视化")


if __name__ == "__main__":
    main() 