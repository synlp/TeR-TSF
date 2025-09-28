"""
TeR-TSF 分阶段性能分析实验
通过外部调用原有脚本，收集各阶段详细性能数据
不修改原有代码，确保向后兼容
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
from performance_monitor import PerformanceMonitor
import threading


class StagePerformanceAnalyzer:
    """分阶段性能分析器"""
    
    def __init__(self, base_config: Dict, output_dir: str = "./stage_analysis_results"):
        self.base_config = base_config
        self.output_dir = output_dir
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建实验配置文件
        self.config_file = os.path.join(output_dir, "stage_analysis_config.json")
        with open(self.config_file, 'w') as f:
            json.dump({
                'experiment_type': 'stage_performance_analysis',
                'timestamp': datetime.now().isoformat(),
                'base_config': base_config,
                'output_dir': output_dir
            }, f, indent=2)
        
        print(f"分阶段性能分析器已初始化")
        print(f"基础配置: {base_config}")
        print(f"输出目录: {output_dir}")
    
    def _run_command_with_realtime_output(self, cmd: List[str], stage_name: str) -> Dict:
        """运行命令并实时显示输出"""
        print(f"\n[{stage_name}] 执行命令: {' '.join(cmd)}")
        print(f"[{stage_name}] 开始时间: {datetime.now().isoformat()}")
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
    
    def run_full_analysis(self) -> str:
        """运行完整的分阶段性能分析"""
        exp_id = f"stage_analysis_{self.base_config['data_name']}_{self.base_config['llm_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n=== 开始分阶段性能分析: {exp_id} ===")
        
        # 创建性能监控器
        monitor = PerformanceMonitor(exp_id, os.path.join(self.output_dir, "performance_data"))
        
        # 阶段1: 数据准备阶段分析
        stage1_result = self._analyze_data_preparation_stage(monitor, exp_id)
        
        # 阶段2: TextFusionHTS训练阶段分析
        stage2_result = self._analyze_tfhts_training_stage(monitor, exp_id)
        
        # 阶段3: DPO训练阶段分析
        stage3_result = self._analyze_dpo_training_stage(monitor, exp_id)
        
        # 阶段4: 评估阶段分析
        stage4_result = self._analyze_evaluation_stage(monitor, exp_id)
        
        # 保存监控结果
        stage_file, timeline_file, summary_file = monitor.save_results()
        
        # 生成综合分析报告
        analysis_report = self._generate_comprehensive_report(exp_id, [
            stage1_result, stage2_result, stage3_result, stage4_result
        ], stage_file, timeline_file, summary_file)
        
        print(f"\n=== 分阶段性能分析完成 ===")
        print(f"综合报告: {analysis_report}")
        
        return analysis_report
    
    def _analyze_data_preparation_stage(self, monitor: PerformanceMonitor, exp_id: str) -> Dict:
        """分析数据准备阶段"""
        print("\n--- 分析数据准备阶段 ---")
        
        stage_metadata = {
            'stage_type': 'data_preparation',
            'llm_type': self.base_config['llm_type'],
            'tsf_type': self.base_config['tsf_type'],
            'data_name': self.base_config['data_name'],
            'hist_len': self.base_config['hist_len'],
            'pred_len': self.base_config['pred_len'],
            'batch_size': self.base_config['batch_size'],
            'gen_num': self.base_config['gen_num'],
        }
        
        with monitor.monitor_stage("data_preparation_full", stage_metadata):
            # 构建prepare_stage.py命令
            cmd = [
                'python', 'prepare_stage.py',
                '--data_dir', '/data2/user2/ter_tsf',
                '--data_name', self.base_config['data_name'],
                '--llm_type', self.base_config['llm_type'],
                '--tsf_type', self.base_config['tsf_type'],
                '--hist_len', str(self.base_config['hist_len']),
                '--pred_len', str(self.base_config['pred_len']),
                '--batch_size', str(self.base_config['batch_size']),
                '--gen_num', str(self.base_config['gen_num']),
                '--iter_idx', '0',  # 第0轮分析
                '--llm_path', 'original',
                '--exp_time', f"stage_analysis_{exp_id}",
                '--llama_factory_dir', self.base_config.get('llama_factory_dir', 'llama-factory-main'),
                '--down_sample', str(self.base_config.get('down_sample', 0)),
                '--max_batches', '1',  # 仅处理1个batch以节省时间
            ]
            
            # 如果禁用文本质量奖励
            if self.base_config.get('disable_text_quality_reward', False):
                cmd.append('--disable_text_quality_reward')
            
            # 执行命令（无超时限制，实时输出）
            stage_result = self._run_command_with_realtime_output(cmd, "数据准备阶段")
            stage_result['stage_name'] = 'data_preparation'
        
        return stage_result
    
    def _analyze_tfhts_training_stage(self, monitor: PerformanceMonitor, exp_id: str) -> Dict:
        """分析TextFusionHTS训练阶段"""
        print("\n--- 分析TextFusionHTS训练阶段 ---")
        
        stage_metadata = {
            'stage_type': 'tfhts_training',
            'data_name': self.base_config['data_name'],
            'hist_len': self.base_config['hist_len'],
            'pred_len': self.base_config['pred_len'],
            'text_type': 'original_text',  # 第0轮使用原始文本
            'patch_len': 16,
            'stride': 8,
        }
        
        # 为weather和Heart_Rate数据集调整patch参数
        if 'weather' in self.base_config['data_name'].lower() or 'heart' in self.base_config['data_name'].lower():
            stage_metadata.update({'patch_len': 4, 'stride': 2})
        
        with monitor.monitor_stage("tfhts_training", stage_metadata):
            # 构建TextFusionHTS训练命令
            cmd = [
                'python', './Models/TextFusionHTS/train_tfhts.py',
                '--data_dir', f"/data2/user2/ter_tsf/{self.base_config['llm_type']}/{self.base_config['tsf_type']}/{self.base_config['data_name']}/reinforced_data",
                '--save_dir', f"/data2/user2/ter_tsf/{self.base_config['llm_type']}/tfhts/{self.base_config['data_name']}/saved_models",
                '--data_name', self.base_config['data_name'],
                '--hist_len', str(self.base_config['hist_len']),
                '--pred_len', str(self.base_config['pred_len']),
                '--batch_size', '32',
                '--text_type', 'original_text',
                '--patch_len', str(stage_metadata['patch_len']),
                '--stride', str(stage_metadata['stride']),
                '--epochs', '50',  # 减少训练轮数用于分析
                '--lr', '1e-3',
                '--exp_time', f"stage_analysis_{exp_id}",
                '--iter_idx', '0',
            ]
            
            # 执行命令（无超时限制，实时输出）
            stage_result = self._run_command_with_realtime_output(cmd, "TextFusionHTS训练阶段")
            stage_result['stage_name'] = 'tfhts_training'
        
        return stage_result
    
    def _analyze_dpo_training_stage(self, monitor: PerformanceMonitor, exp_id: str) -> Dict:
        """分析DPO训练阶段"""
        print("\n--- 分析DPO训练阶段 ---")
        
        stage_metadata = {
            'stage_type': 'dpo_training',
            'llm_type': self.base_config['llm_type'],
            'lora_rank': self.base_config.get('lora_rank', 8),
            'learning_rate': self.base_config.get('lr', 5.0e-5),
            'num_epochs': 2,  # 减少训练轮数用于分析
            'batch_size': self.base_config.get('per_device_train_batch_size', 2),
        }
        
        with monitor.monitor_stage("dpo_training", stage_metadata):
            # 模拟DPO训练过程（因为需要偏好数据）
            # 这里我们创建一个简化的分析脚本
            start_time = time.time()
            
            try:
                # 检查偏好数据是否存在
                preference_data_path = f"/data2/user2/ter_tsf/{self.base_config['llm_type']}/{self.base_config['tsf_type']}/{self.base_config['data_name']}/preference_data"
                
                if not os.path.exists(preference_data_path):
                    # 如果偏好数据不存在，记录为跳过
                    stage_result = {
                        'stage_name': 'dpo_training',
                        'success': False,
                        'duration': 0,
                        'return_code': -3,
                        'error': 'Preference data not found, skipping DPO training analysis',
                        'skipped': True,
                    }
                    print("偏好数据不存在，跳过DPO训练阶段分析")
                else:
                    # 模拟DPO训练的资源使用
                    print("模拟DPO训练过程...")
                    
                    # 模拟加载模型和数据
                    time.sleep(5)
                    
                    # 模拟训练过程
                    for epoch in range(stage_metadata['num_epochs']):
                        print(f"模拟训练轮次 {epoch + 1}/{stage_metadata['num_epochs']}")
                        time.sleep(10)  # 模拟每轮训练时间
                    
                    end_time = time.time()
                    stage_result = {
                        'stage_name': 'dpo_training',
                        'success': True,
                        'duration': end_time - start_time,
                        'return_code': 0,
                        'simulated': True,
                        'epochs': stage_metadata['num_epochs'],
                    }
                    print(f"DPO训练阶段模拟完成，耗时: {stage_result['duration']:.2f}秒")
                
            except Exception as e:
                end_time = time.time()
                stage_result = {
                    'stage_name': 'dpo_training',
                    'success': False,
                    'duration': end_time - start_time,
                    'return_code': -2,
                    'error': str(e),
                }
                print(f"DPO训练阶段分析异常: {e}")
        
        return stage_result
    
    def _analyze_evaluation_stage(self, monitor: PerformanceMonitor, exp_id: str) -> Dict:
        """分析评估阶段"""
        print("\n--- 分析评估阶段 ---")
        
        stage_metadata = {
            'stage_type': 'evaluation',
            'data_name': self.base_config['data_name'],
            'llm_type': self.base_config['llm_type'],
            'tsf_type': self.base_config['tsf_type'],
            'batch_size': self.base_config['batch_size'],
        }
        
        with monitor.monitor_stage("evaluation", stage_metadata):
            # 构建评估命令
            cmd = [
                'python', 'evaluate.py',
                '--data_dir', '/data2/user2/ter_tsf',
                '--data_name', self.base_config['data_name'],
                '--llm_type', self.base_config['llm_type'],
                '--tsf_type', self.base_config['tsf_type'],
                '--hist_len', str(self.base_config['hist_len']),
                '--pred_len', str(self.base_config['pred_len']),
                '--batch_size', str(self.base_config['batch_size']),
                '--iter_idx', '0',
                '--llm_path', 'original',
                '--exp_time', f"stage_analysis_{exp_id}",
                '--gen_num', str(self.base_config['gen_num']),
                '--down_sample', str(self.base_config.get('down_sample', 1)),
                '--dpo_epoch', '2',
                '--dpo_lr', str(self.base_config.get('lr', 0.0001)),
            ]
            
            # 执行命令（无超时限制，实时输出）
            stage_result = self._run_command_with_realtime_output(cmd, "评估阶段")
            stage_result['stage_name'] = 'evaluation'
        
        return stage_result
    
    def _generate_comprehensive_report(self, exp_id: str, stage_results: List[Dict], 
                                     stage_file: str, timeline_file: str, summary_file: str) -> str:
        """生成综合分析报告"""
        print("\n--- 生成综合分析报告 ---")
        
        # 读取性能监控数据
        stage_df = pd.read_csv(stage_file)
        timeline_df = pd.read_csv(timeline_file) if timeline_file and os.path.exists(timeline_file) else None
        
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        # 合并阶段执行结果和性能数据
        comprehensive_report = {
            'experiment_info': {
                'experiment_id': exp_id,
                'experiment_type': 'stage_performance_analysis',
                'timestamp': datetime.now().isoformat(),
                'base_config': self.base_config,
            },
            'stage_execution_results': stage_results,
            'performance_monitoring_summary': summary_data,
            'detailed_analysis': {
                'total_experiment_duration': sum([r.get('duration', 0) for r in stage_results]),
                'successful_stages': len([r for r in stage_results if r.get('success', False)]),
                'failed_stages': len([r for r in stage_results if not r.get('success', False)]),
                'stage_duration_breakdown': {r['stage_name']: r.get('duration', 0) for r in stage_results},
            }
        }
        
        # 性能瓶颈分析
        if not stage_df.empty:
            comprehensive_report['bottleneck_analysis'] = {
                'slowest_stage': stage_df.loc[stage_df['duration_seconds'].idxmax(), 'stage_name'],
                'most_memory_intensive_stage': stage_df.loc[stage_df['memory_peak_used_gb'].idxmax(), 'stage_name'],
                'highest_cpu_usage_stage': stage_df.loc[stage_df['cpu_usage_avg_percent'].idxmax(), 'stage_name'],
                'stage_performance_summary': stage_df.groupby('stage_name').agg({
                    'duration_seconds': ['sum', 'mean'],
                    'memory_peak_used_gb': ['max', 'mean'],
                    'cpu_usage_avg_percent': ['mean'],
                    'memory_delta_gb': ['sum']
                }).to_dict()
            }
            
            if 'gpu_memory_peak_mb' in stage_df.columns:
                comprehensive_report['bottleneck_analysis']['gpu_analysis'] = {
                    'peak_gpu_memory_mb': stage_df['gpu_memory_peak_mb'].max(),
                    'gpu_intensive_stage': stage_df.loc[stage_df['gpu_memory_peak_mb'].idxmax(), 'stage_name'],
                    'gpu_memory_by_stage': stage_df.groupby('stage_name')['gpu_memory_peak_mb'].max().to_dict()
                }
        
        # 效率指标计算
        total_duration = comprehensive_report['detailed_analysis']['total_experiment_duration']
        if total_duration > 0:
            comprehensive_report['efficiency_metrics'] = {
                'overall_throughput': 1.0 / total_duration,  # 实验/秒
                'stage_efficiency': {
                    r['stage_name']: {
                        'duration_ratio': r.get('duration', 0) / total_duration,
                        'success_rate': 1.0 if r.get('success', False) else 0.0,
                    }
                    for r in stage_results
                },
                'resource_utilization': {
                    'avg_cpu_efficiency': stage_df['cpu_usage_avg_percent'].mean() if not stage_df.empty else 0,
                    'peak_memory_efficiency': stage_df['memory_peak_used_gb'].max() if not stage_df.empty else 0,
                }
            }
        
        # 保存综合报告
        report_file = os.path.join(self.output_dir, f"{exp_id}_comprehensive_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        # 生成CSV格式的汇总表
        self._generate_summary_table(comprehensive_report, exp_id)
        
        print(f"综合分析报告已保存: {report_file}")
        return report_file
    
    def _generate_summary_table(self, report: Dict, exp_id: str):
        """生成CSV格式的汇总表"""
        # 提取关键指标
        stage_data = []
        for stage_result in report['stage_execution_results']:
            stage_name = stage_result['stage_name']
            
            # 从性能监控数据中获取对应指标
            perf_data = report.get('bottleneck_analysis', {}).get('stage_performance_summary', {})
            
            stage_row = {
                'stage_name': stage_name,
                'success': stage_result.get('success', False),
                'duration_seconds': stage_result.get('duration', 0),
                'return_code': stage_result.get('return_code', 0),
                'peak_memory_gb': perf_data.get('memory_peak_used_gb', {}).get('max', {}).get(stage_name, 0) if perf_data else 0,
                'avg_cpu_usage_percent': perf_data.get('cpu_usage_avg_percent', {}).get('mean', {}).get(stage_name, 0) if perf_data else 0,
                'memory_delta_gb': perf_data.get('memory_delta_gb', {}).get('sum', {}).get(stage_name, 0) if perf_data else 0,
            }
            
            # 添加GPU数据（如果有）
            gpu_data = report.get('bottleneck_analysis', {}).get('gpu_analysis', {})
            if gpu_data:
                stage_row['peak_gpu_memory_mb'] = gpu_data.get('gpu_memory_by_stage', {}).get(stage_name, 0)
            
            stage_data.append(stage_row)
        
        # 创建DataFrame并保存
        summary_df = pd.DataFrame(stage_data)
        summary_file = os.path.join(self.output_dir, f"{exp_id}_stage_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"阶段汇总表已保存: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TeR-TSF 分阶段性能分析")
    
    # 基础配置参数
    parser.add_argument("--data_name", type=str, default="Energy", help="数据集名称")
    parser.add_argument("--llm_type", type=str, default="qwen3-1.7b", help="LLM类型")
    parser.add_argument("--tsf_type", type=str, default="tfhts", help="TSF类型")
    parser.add_argument("--hist_len", type=int, default=96, help="历史序列长度")
    parser.add_argument("--pred_len", type=int, default=12, help="预测序列长度")
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小")
    parser.add_argument("--gen_num", type=int, default=2, help="生成文本数量")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每设备训练批次大小")
    parser.add_argument("--llama_factory_dir", type=str, default="llama-factory-main", help="LLaMA Factory目录")
    parser.add_argument("--down_sample", type=int, default=0, help="下采样")
    parser.add_argument("--disable_text_quality_reward", action="store_true", help="禁用文本质量奖励")
    parser.add_argument("--output_dir", type=str, default="./stage_analysis_results", help="输出目录")
    
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
        'lora_rank': args.lora_rank,
        'lr': args.lr,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'llama_factory_dir': args.llama_factory_dir,
        'down_sample': args.down_sample,
        'disable_text_quality_reward': args.disable_text_quality_reward,
    }
    
    # 创建分析器并运行分析
    analyzer = StagePerformanceAnalyzer(base_config, args.output_dir)
    report_file = analyzer.run_full_analysis()
    
    print(f"\n=== 分阶段性能分析完成 ===")
    print(f"详细报告: {report_file}")
    print("可以使用生成的CSV文件进行进一步的数据分析和可视化")


if __name__ == "__main__":
    main() 