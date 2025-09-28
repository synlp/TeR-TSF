#!/usr/bin/env python3
"""
reinforced_text后处理程序
实现混合型轻量级处理，优化文本以提升时间序列预测效果
"""

import pandas as pd
import numpy as np
import re
import argparse
from typing import List, Tuple

def extract_numerical_insights(history_series: List[float]) -> str:
    """基于历史数据提取数值洞察"""
    data = np.array(history_series)
    
    # 基本统计特征
    mean_val = np.mean(data)
    std_val = np.std(data)
    trend = "increasing" if data[-1] > data[0] else "decreasing"
    volatility = "high" if std_val > mean_val * 0.2 else "moderate" if std_val > mean_val * 0.1 else "low"
    
    # 趋势分析
    recent_trend = "upward" if np.mean(data[-6:]) > np.mean(data[-12:-6]) else "downward"
    
    # 范围分析
    min_val, max_val = np.min(data), np.max(data)
    current_level = "high" if data[-1] > mean_val + std_val else "low" if data[-1] < mean_val - std_val else "normal"
    
    insights = f"Historical analysis: Mean={mean_val:.1f}, volatility={volatility}. " \
              f"Overall {trend} trend with recent {recent_trend} movement. " \
              f"Current level is {current_level} (range: {min_val:.1f}-{max_val:.1f}). " \
              f"Prediction should consider {volatility} volatility patterns."
    
    return insights

def optimize_text_content(text: str) -> str:
    """优化文本内容，提取核心预测信息"""
    
    # 移除冗余的理论描述
    text = re.sub(r'\*\*\d+\.\s+[^*]+\*\*', '', text)  # 移除编号标题
    text = re.sub(r'The task is.*?done\s*done', '', text, flags=re.DOTALL)  # 移除任务描述
    text = re.sub(r'Answer:\s*', '', text)  # 移除Answer标记
    
    # 保留关键预测相关段落
    key_phrases = [
        r'[Ss]easonal[^.]*\.',
        r'[Tt]rend[^.]*\.',
        r'[Pp]attern[^.]*\.',
        r'[Ff]orecast[^.]*\.',
        r'[Pp]recipitation[^.]*\.',
        r'[Tt]emperature[^.]*\.',
        r'[Ww]eather[^.]*\.'
    ]
    
    key_content = []
    for phrase_pattern in key_phrases:
        matches = re.findall(phrase_pattern, text)
        key_content.extend(matches[:2])  # 每种类型最多保留2句
    
    # 如果关键内容不足，保留原文前部分
    if len(' '.join(key_content)) < 300:
        sentences = re.split(r'[.!?]+', text)
        key_content = sentences[:5]  # 保留前5句
    
    optimized_text = ' '.join(key_content)
    
    # 清理多余空格和换行
    optimized_text = re.sub(r'\s+', ' ', optimized_text).strip()
    
    return optimized_text

def postprocess_reinforced_text(history_series: List[float], original_text: str) -> str:
    """混合型后处理：结合数值洞察和语义优化"""
    
    # 1. 提取数值洞察
    numerical_insights = extract_numerical_insights(history_series)
    
    # 2. 优化原始文本
    optimized_content = optimize_text_content(original_text)
    
    # 3. 构建混合文本
    enhanced_text = f"{numerical_insights} {optimized_content}"
    
    # 4. 长度控制（确保在1500字符以内）
    if len(enhanced_text) > 1500:
        # 智能截取，优先保留数值洞察
        if len(numerical_insights) < 1200:
            remaining_space = 1500 - len(numerical_insights) - 10
            optimized_content = optimized_content[:remaining_space] + "..."
        enhanced_text = f"{numerical_insights} {optimized_content}"
    
    return enhanced_text.strip()

def process_csv_file(input_file: str, output_file: str):
    """处理整个CSV文件"""
    print(f"读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"开始处理 {len(df)} 行数据...")
    processed_texts = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(df)}")
        
        # 解析历史序列数据
        history_series = eval(row['history_series'])
        original_text = str(row['reinforced_text'])
        
        # 后处理文本
        enhanced_text = postprocess_reinforced_text(history_series, original_text)
        processed_texts.append(enhanced_text)
    
    # 创建新的数据框
    df_new = df.copy()
    df_new['reinforced_text'] = processed_texts
    
    # 保存结果
    df_new.to_csv(output_file, index=False)
    print(f"处理完成，已保存到: {output_file}")
    
    # 输出统计信息
    original_avg_len = df['reinforced_text'].str.len().mean()
    new_avg_len = df_new['reinforced_text'].str.len().mean()
    print(f"平均文本长度: {original_avg_len:.0f} -> {new_avg_len:.0f} 字符")

def main():
    parser = argparse.ArgumentParser(description="reinforced_text后处理程序")
    parser.add_argument('--input_file', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出CSV文件路径')
    
    args = parser.parse_args()
    process_csv_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 