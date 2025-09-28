"""
TeR-TSF 性能实验结果分析脚本
用于处理和汇总分阶段性能分析和参数敏感性分析的结果
生成用于论文的结构化数据表格
"""

import pandas as pd
import numpy as np
import json
import os
import glob
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class PerformanceResultsAnalyzer:
    """性能实验结果分析器"""
    
    def __init__(self, results_dir: str, output_dir: str = None):
        self.results_dir = results_dir
        self.output_dir = output_dir or os.path.join(results_dir, "analysis_output")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"性能结果分析器已初始化")
        print(f"结果目录: {results_dir}")
        print(f"输出目录: {self.output_dir}")
    
    def analyze_all_results(self) -> Dict:
        """分析所有实验结果"""
        print("\n=== 开始分析所有实验结果 ===")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'results_directory': self.results_dir,
            'stage_analysis': None,
            'parameter_sensitivity': None,
            'comparative_analysis': None,
        }
        
        # 分析分阶段性能结果
        stage_analysis = self.analyze_stage_performance()
        if stage_analysis:
            analysis_results['stage_analysis'] = stage_analysis
        
        # 分析参数敏感性结果
        param_analysis = self.analyze_parameter_sensitivity()
        if param_analysis:
            analysis_results['parameter_sensitivity'] = param_analysis
        
        # 生成对比分析
        if stage_analysis and param_analysis:
            comparative_analysis = self.generate_comparative_analysis(stage_analysis, param_analysis)
            analysis_results['comparative_analysis'] = comparative_analysis
        
        # 保存综合分析结果
        self._save_comprehensive_analysis(analysis_results)
        
        return analysis_results
    
    def analyze_stage_performance(self) -> Optional[Dict]:
        """分析分阶段性能结果"""
        print("\n--- 分析分阶段性能结果 ---")
        
        # 查找分阶段分析结果文件
        stage_files = self._find_stage_analysis_files()
        if not stage_files:
            print("未找到分阶段分析结果文件")
            return None
        
        print(f"找到 {len(stage_files)} 个分阶段分析文件")
        
        # 合并所有阶段数据
        all_stage_data = []
        for file_path in stage_files:
            try:
                df = pd.read_csv(file_path)
                all_stage_data.append(df)
                print(f"加载文件: {os.path.basename(file_path)} ({len(df)} 条记录)")
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        if not all_stage_data:
            return None
        
        # 合并数据
        combined_df = pd.concat(all_stage_data, ignore_index=True)
        
        # 生成阶段分析统计
        stage_analysis = self._analyze_stage_statistics(combined_df)
        
        # 保存阶段分析结果
        self._save_stage_analysis_results(stage_analysis, combined_df)
        
        return stage_analysis
    
    def analyze_parameter_sensitivity(self) -> Optional[Dict]:
        """分析参数敏感性结果"""
        print("\n--- 分析参数敏感性结果 ---")
        
        # 查找参数敏感性分析结果文件
        param_files = self._find_parameter_sensitivity_files()
        if not param_files:
            print("未找到参数敏感性分析结果文件")
            return None
        
        print(f"找到 {len(param_files)} 个参数敏感性分析文件")
        
        # 加载参数敏感性数据
        param_data = []
        for file_path in param_files:
            try:
                df = pd.read_csv(file_path)
                param_data.append(df)
                print(f"加载文件: {os.path.basename(file_path)} ({len(df)} 条记录)")
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        if not param_data:
            return None
        
        # 合并数据
        combined_param_df = pd.concat(param_data, ignore_index=True)
        
        # 生成参数敏感性分析
        param_analysis = self._analyze_parameter_statistics(combined_param_df)
        
        # 保存参数敏感性分析结果
        self._save_parameter_analysis_results(param_analysis, combined_param_df)
        
        return param_analysis
    
    def _find_stage_analysis_files(self) -> List[str]:
        """查找分阶段分析结果文件"""
        patterns = [
            os.path.join(self.results_dir, "**/stage_analysis/**/performance_data/*_stage_stats_*.csv"),
            os.path.join(self.results_dir, "**/stage_analysis/**/*_stage_summary.csv"),
            os.path.join(self.results_dir, "**/*stage*stats*.csv"),
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))
        
        # 去重并排序
        return sorted(list(set(files)))
    
    def _find_parameter_sensitivity_files(self) -> List[str]:
        """查找参数敏感性分析结果文件"""
        patterns = [
            os.path.join(self.results_dir, "**/param_sensitivity/**/analysis_results_*.csv"),
            os.path.join(self.results_dir, "**/param_sensitivity/**/*_comprehensive_report.json"),
            os.path.join(self.results_dir, "**/*param*analysis*.csv"),
        ]
        
        files = []
        for pattern in patterns:
            found_files = glob.glob(pattern, recursive=True)
            # 只保留CSV文件用于数据分析
            files.extend([f for f in found_files if f.endswith('.csv')])
        
        return sorted(list(set(files)))
    
    def _analyze_stage_statistics(self, df: pd.DataFrame) -> Dict:
        """分析阶段统计数据"""
        print("计算阶段统计指标...")
        
        analysis = {
            'total_experiments': len(df['experiment_name'].unique()) if 'experiment_name' in df.columns else 1,
            'total_stages': len(df),
            'unique_stage_types': df['stage_name'].unique().tolist() if 'stage_name' in df.columns else [],
            'duration_statistics': {},
            'memory_statistics': {},
            'cpu_statistics': {},
            'gpu_statistics': {},
            'stage_breakdown': {},
            'bottleneck_analysis': {},
        }
        
        # 持续时间统计
        if 'duration_seconds' in df.columns:
            analysis['duration_statistics'] = {
                'total_duration_seconds': df['duration_seconds'].sum(),
                'mean_duration_seconds': df['duration_seconds'].mean(),
                'std_duration_seconds': df['duration_seconds'].std(),
                'min_duration_seconds': df['duration_seconds'].min(),
                'max_duration_seconds': df['duration_seconds'].max(),
                'median_duration_seconds': df['duration_seconds'].median(),
            }
        
        # 内存使用统计
        memory_cols = [col for col in df.columns if 'memory' in col.lower()]
        if memory_cols:
            for col in memory_cols:
                if df[col].dtype in ['float64', 'int64']:
                    analysis['memory_statistics'][col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'median': df[col].median(),
                    }
        
        # CPU使用统计
        cpu_cols = [col for col in df.columns if 'cpu' in col.lower()]
        if cpu_cols:
            for col in cpu_cols:
                if df[col].dtype in ['float64', 'int64']:
                    analysis['cpu_statistics'][col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'median': df[col].median(),
                    }
        
        # GPU使用统计
        gpu_cols = [col for col in df.columns if 'gpu' in col.lower()]
        if gpu_cols:
            for col in gpu_cols:
                if df[col].dtype in ['float64', 'int64']:
                    analysis['gpu_statistics'][col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'median': df[col].median(),
                    }
        
        # 按阶段分解分析
        if 'stage_name' in df.columns:
            stage_groups = df.groupby('stage_name')
            
            for stage_name, stage_data in stage_groups:
                stage_stats = {
                    'count': len(stage_data),
                    'avg_duration': stage_data['duration_seconds'].mean() if 'duration_seconds' in stage_data.columns else 0,
                    'total_duration': stage_data['duration_seconds'].sum() if 'duration_seconds' in stage_data.columns else 0,
                }
                
                # 添加内存和CPU统计
                if 'memory_peak_used_gb' in stage_data.columns:
                    stage_stats['avg_memory_gb'] = stage_data['memory_peak_used_gb'].mean()
                    stage_stats['max_memory_gb'] = stage_data['memory_peak_used_gb'].max()
                
                if 'cpu_usage_avg_percent' in stage_data.columns:
                    stage_stats['avg_cpu_percent'] = stage_data['cpu_usage_avg_percent'].mean()
                
                analysis['stage_breakdown'][stage_name] = stage_stats
        
        # 瓶颈分析
        if 'duration_seconds' in df.columns and 'stage_name' in df.columns:
            slowest_stage_idx = df['duration_seconds'].idxmax()
            analysis['bottleneck_analysis']['slowest_stage'] = {
                'stage_name': df.loc[slowest_stage_idx, 'stage_name'],
                'duration_seconds': df.loc[slowest_stage_idx, 'duration_seconds'],
            }
        
        if 'memory_peak_used_gb' in df.columns and 'stage_name' in df.columns:
            memory_intensive_idx = df['memory_peak_used_gb'].idxmax()
            analysis['bottleneck_analysis']['memory_intensive_stage'] = {
                'stage_name': df.loc[memory_intensive_idx, 'stage_name'],
                'memory_gb': df.loc[memory_intensive_idx, 'memory_peak_used_gb'],
            }
        
        return analysis
    
    def _analyze_parameter_statistics(self, df: pd.DataFrame) -> Dict:
        """分析参数敏感性统计数据"""
        print("计算参数敏感性统计指标...")
        
        analysis = {
            'total_experiments': len(df),
            'parameters_tested': [],
            'parameter_ranges': {},
            'performance_correlations': {},
            'parameter_impact_ranking': [],
            'optimal_configurations': {},
            'scalability_analysis': {},
        }
        
        # 识别测试的参数
        if 'variable_param' in df.columns:
            analysis['parameters_tested'] = df['variable_param'].unique().tolist()
            
            # 分析每个参数的影响
            for param_name in analysis['parameters_tested']:
                param_data = df[df['variable_param'] == param_name].copy()
                param_data = param_data.sort_values('variable_value')
                
                if len(param_data) > 1:
                    # 参数范围
                    analysis['parameter_ranges'][param_name] = {
                        'min': param_data['variable_value'].min(),
                        'max': param_data['variable_value'].max(),
                        'values': param_data['variable_value'].unique().tolist(),
                        'count': len(param_data),
                    }
                    
                    # 性能相关性分析
                    correlations = {}
                    performance_metrics = ['total_duration', 'perf_peak_memory_gb', 'perf_throughput', 'perf_memory_efficiency']
                    
                    for metric in performance_metrics:
                        if metric in param_data.columns:
                            corr = param_data[['variable_value', metric]].corr().iloc[0, 1]
                            if not np.isnan(corr):
                                correlations[metric] = corr
                    
                    analysis['performance_correlations'][param_name] = correlations
                    
                    # 计算参数影响程度
                    if 'total_duration' in param_data.columns:
                        duration_range = param_data['total_duration'].max() - param_data['total_duration'].min()
                        duration_mean = param_data['total_duration'].mean()
                        impact_score = duration_range / duration_mean if duration_mean > 0 else 0
                        
                        analysis['parameter_impact_ranking'].append({
                            'parameter': param_name,
                            'impact_score': impact_score,
                            'duration_range': duration_range,
                            'duration_mean': duration_mean,
                        })
                    
                    # 最优配置分析
                    if 'perf_memory_efficiency' in param_data.columns:
                        best_idx = param_data['perf_memory_efficiency'].idxmax()
                        analysis['optimal_configurations'][param_name] = {
                            'best_value': param_data.loc[best_idx, 'variable_value'],
                            'best_efficiency': param_data.loc[best_idx, 'perf_memory_efficiency'],
                            'best_duration': param_data.loc[best_idx, 'total_duration'],
                            'best_memory': param_data.loc[best_idx, 'perf_peak_memory_gb'] if 'perf_peak_memory_gb' in param_data.columns else 0,
                        }
                    elif 'total_duration' in param_data.columns:
                        best_idx = param_data['total_duration'].idxmin()
                        analysis['optimal_configurations'][param_name] = {
                            'best_value': param_data.loc[best_idx, 'variable_value'],
                            'best_duration': param_data.loc[best_idx, 'total_duration'],
                            'best_memory': param_data.loc[best_idx, 'perf_peak_memory_gb'] if 'perf_peak_memory_gb' in param_data.columns else 0,
                        }
                    
                    # 可扩展性分析
                    scalability_metrics = {}
                    if 'total_duration' in param_data.columns:
                        # 线性拟合分析
                        x = param_data['variable_value'].values
                        y = param_data['total_duration'].values
                        
                        if len(x) > 1:
                            # 计算线性相关系数
                            correlation = np.corrcoef(x, y)[0, 1] if not np.isnan(np.corrcoef(x, y)[0, 1]) else 0
                            scalability_metrics['duration_linearity'] = abs(correlation)
                            
                            # 计算增长率
                            if y[0] > 0:
                                growth_rate = (y[-1] - y[0]) / y[0]
                                scalability_metrics['duration_growth_rate'] = growth_rate
                    
                    analysis['scalability_analysis'][param_name] = scalability_metrics
        
        # 按影响程度排序
        analysis['parameter_impact_ranking'] = sorted(
            analysis['parameter_impact_ranking'], 
            key=lambda x: x['impact_score'], 
            reverse=True
        )
        
        return analysis
    
    def generate_comparative_analysis(self, stage_analysis: Dict, param_analysis: Dict) -> Dict:
        """生成对比分析"""
        print("生成对比分析...")
        
        comparative = {
            'experiment_comparison': {},
            'performance_insights': [],
            'optimization_recommendations': [],
            'bottleneck_summary': {},
        }
        
        # 实验对比
        comparative['experiment_comparison'] = {
            'stage_analysis': {
                'total_stages': stage_analysis.get('total_stages', 0),
                'total_duration': stage_analysis.get('duration_statistics', {}).get('total_duration_seconds', 0),
                'bottleneck_stage': stage_analysis.get('bottleneck_analysis', {}).get('slowest_stage', {}).get('stage_name', 'unknown'),
            },
            'parameter_analysis': {
                'total_experiments': param_analysis.get('total_experiments', 0),
                'parameters_tested': len(param_analysis.get('parameters_tested', [])),
                'most_impactful_param': param_analysis.get('parameter_impact_ranking', [{}])[0].get('parameter', 'unknown') if param_analysis.get('parameter_impact_ranking') else 'unknown',
            }
        }
        
        # 性能洞察
        insights = []
        
        # 从阶段分析获得洞察
        if stage_analysis.get('bottleneck_analysis', {}).get('slowest_stage'):
            slowest_stage = stage_analysis['bottleneck_analysis']['slowest_stage']
            insights.append(f"性能瓶颈识别: {slowest_stage['stage_name']} 阶段耗时最长 ({slowest_stage['duration_seconds']:.2f}秒)")
        
        # 从参数分析获得洞察
        if param_analysis.get('parameter_impact_ranking'):
            top_impact = param_analysis['parameter_impact_ranking'][0]
            insights.append(f"关键参数识别: {top_impact['parameter']} 对性能影响最大 (影响分数: {top_impact['impact_score']:.3f})")
        
        comparative['performance_insights'] = insights
        
        # 优化建议
        recommendations = []
        
        # 基于阶段分析的建议
        stage_breakdown = stage_analysis.get('stage_breakdown', {})
        if stage_breakdown:
            # 找出耗时最长的阶段
            max_duration_stage = max(stage_breakdown.items(), key=lambda x: x[1].get('total_duration', 0))
            recommendations.append(f"优先优化 {max_duration_stage[0]} 阶段，可获得最大性能提升")
        
        # 基于参数分析的建议
        optimal_configs = param_analysis.get('optimal_configurations', {})
        for param, config in optimal_configs.items():
            recommendations.append(f"参数 {param} 建议设置为 {config.get('best_value')} 以获得最佳效率")
        
        comparative['optimization_recommendations'] = recommendations
        
        # 瓶颈汇总
        comparative['bottleneck_summary'] = {
            'computational_bottleneck': stage_analysis.get('bottleneck_analysis', {}).get('slowest_stage', {}).get('stage_name', 'unknown'),
            'memory_bottleneck': stage_analysis.get('bottleneck_analysis', {}).get('memory_intensive_stage', {}).get('stage_name', 'unknown'),
            'parameter_bottleneck': param_analysis.get('parameter_impact_ranking', [{}])[0].get('parameter', 'unknown') if param_analysis.get('parameter_impact_ranking') else 'unknown',
        }
        
        return comparative
    
    def _save_stage_analysis_results(self, analysis: Dict, df: pd.DataFrame):
        """保存阶段分析结果"""
        # 保存分析报告
        analysis_file = os.path.join(self.output_dir, "stage_analysis_report.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 保存原始数据
        data_file = os.path.join(self.output_dir, "stage_analysis_data.csv")
        df.to_csv(data_file, index=False)
        
        # 生成阶段汇总表
        if analysis.get('stage_breakdown'):
            stage_summary = []
            for stage_name, stats in analysis['stage_breakdown'].items():
                row = {'stage_name': stage_name}
                row.update(stats)
                stage_summary.append(row)
            
            summary_df = pd.DataFrame(stage_summary)
            summary_file = os.path.join(self.output_dir, "stage_performance_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            
            print(f"阶段分析结果已保存:")
            print(f"  - 分析报告: {analysis_file}")
            print(f"  - 原始数据: {data_file}")
            print(f"  - 汇总表格: {summary_file}")
    
    def _save_parameter_analysis_results(self, analysis: Dict, df: pd.DataFrame):
        """保存参数分析结果"""
        # 保存分析报告
        analysis_file = os.path.join(self.output_dir, "parameter_analysis_report.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 保存原始数据
        data_file = os.path.join(self.output_dir, "parameter_analysis_data.csv")
        df.to_csv(data_file, index=False)
        
        # 生成参数影响汇总表
        if analysis.get('parameter_impact_ranking'):
            impact_df = pd.DataFrame(analysis['parameter_impact_ranking'])
            impact_file = os.path.join(self.output_dir, "parameter_impact_ranking.csv")
            impact_df.to_csv(impact_file, index=False)
            
            # 生成最优配置表
            if analysis.get('optimal_configurations'):
                optimal_configs = []
                for param, config in analysis['optimal_configurations'].items():
                    row = {'parameter': param}
                    row.update(config)
                    optimal_configs.append(row)
                
                optimal_df = pd.DataFrame(optimal_configs)
                optimal_file = os.path.join(self.output_dir, "optimal_configurations.csv")
                optimal_df.to_csv(optimal_file, index=False)
                
                print(f"参数分析结果已保存:")
                print(f"  - 分析报告: {analysis_file}")
                print(f"  - 原始数据: {data_file}")
                print(f"  - 影响排序: {impact_file}")
                print(f"  - 最优配置: {optimal_file}")
    
    def _save_comprehensive_analysis(self, analysis: Dict):
        """保存综合分析结果"""
        # 保存完整分析报告
        comprehensive_file = os.path.join(self.output_dir, "comprehensive_analysis_report.json")
        with open(comprehensive_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 生成论文用数据表格
        self._generate_paper_tables(analysis)
        
        print(f"综合分析报告已保存: {comprehensive_file}")
    
    def _generate_paper_tables(self, analysis: Dict):
        """生成论文用数据表格"""
        print("生成论文用数据表格...")
        
        tables_dir = os.path.join(self.output_dir, "paper_tables")
        os.makedirs(tables_dir, exist_ok=True)
        
        # 表格1: 阶段性能汇总
        if analysis.get('stage_analysis', {}).get('stage_breakdown'):
            stage_data = []
            for stage_name, stats in analysis['stage_analysis']['stage_breakdown'].items():
                stage_data.append({
                    'Stage': stage_name.replace('_', ' ').title(),
                    'Count': stats.get('count', 0),
                    'Avg Duration (s)': f"{stats.get('avg_duration', 0):.2f}",
                    'Total Duration (s)': f"{stats.get('total_duration', 0):.2f}",
                    'Avg Memory (GB)': f"{stats.get('avg_memory_gb', 0):.2f}",
                    'Max Memory (GB)': f"{stats.get('max_memory_gb', 0):.2f}",
                    'Avg CPU (%)': f"{stats.get('avg_cpu_percent', 0):.1f}",
                })
            
            stage_table_df = pd.DataFrame(stage_data)
            stage_table_file = os.path.join(tables_dir, "Table1_Stage_Performance.csv")
            stage_table_df.to_csv(stage_table_file, index=False)
        
        # 表格2: 参数影响排序
        if analysis.get('parameter_sensitivity', {}).get('parameter_impact_ranking'):
            param_data = []
            for param_info in analysis['parameter_sensitivity']['parameter_impact_ranking']:
                param_data.append({
                    'Parameter': param_info['parameter'].replace('_', ' ').title(),
                    'Impact Score': f"{param_info['impact_score']:.3f}",
                    'Duration Range (s)': f"{param_info['duration_range']:.2f}",
                    'Avg Duration (s)': f"{param_info['duration_mean']:.2f}",
                })
            
            param_table_df = pd.DataFrame(param_data)
            param_table_file = os.path.join(tables_dir, "Table2_Parameter_Impact.csv")
            param_table_df.to_csv(param_table_file, index=False)
        
        # 表格3: 最优配置
        if analysis.get('parameter_sensitivity', {}).get('optimal_configurations'):
            optimal_data = []
            for param, config in analysis['parameter_sensitivity']['optimal_configurations'].items():
                optimal_data.append({
                    'Parameter': param.replace('_', ' ').title(),
                    'Optimal Value': config.get('best_value', 'N/A'),
                    'Best Duration (s)': f"{config.get('best_duration', 0):.2f}",
                    'Best Memory (GB)': f"{config.get('best_memory', 0):.2f}",
                    'Efficiency Score': f"{config.get('best_efficiency', 0):.3f}" if 'best_efficiency' in config else 'N/A',
                })
            
            optimal_table_df = pd.DataFrame(optimal_data)
            optimal_table_file = os.path.join(tables_dir, "Table3_Optimal_Configurations.csv")
            optimal_table_df.to_csv(optimal_table_file, index=False)
        
        print(f"论文用数据表格已保存到: {tables_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TeR-TSF 性能实验结果分析")
    parser.add_argument("--results_dir", required=True, help="实验结果目录")
    parser.add_argument("--output_dir", help="分析输出目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"错误：结果目录不存在: {args.results_dir}")
        return
    
    # 创建分析器
    analyzer = PerformanceResultsAnalyzer(args.results_dir, args.output_dir)
    
    # 执行分析
    results = analyzer.analyze_all_results()
    
    print(f"\n=== 性能结果分析完成 ===")
    print(f"分析结果已保存到: {analyzer.output_dir}")
    print("主要输出文件:")
    print("  - comprehensive_analysis_report.json: 完整分析报告")
    print("  - stage_analysis_report.json: 分阶段分析报告")
    print("  - parameter_analysis_report.json: 参数敏感性分析报告")
    print("  - paper_tables/: 论文用数据表格")


if __name__ == "__main__":
    main() 