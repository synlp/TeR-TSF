#!/usr/bin/env python3
"""
测试实时监控功能的简单脚本
模拟长时间运行的实验过程
"""

import time
import random
import os
import sys
import torch

def simulate_data_preparation():
    """模拟数据准备阶段"""
    print("开始数据准备阶段...")
    for i in range(10):
        # 模拟一些计算
        data = [random.random() for _ in range(100000)]
        time.sleep(2)
        print(f"数据准备进度: {(i+1)*10}%")
    print("数据准备完成!")

def simulate_model_training():
    """模拟模型训练阶段"""
    print("开始模型训练阶段...")
    for epoch in range(5):
        # 模拟GPU使用
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x.t())
            del x, y
            torch.cuda.empty_cache()
        
        time.sleep(3)
        loss = random.uniform(0.5, 2.0) * (0.8 ** epoch)  # 模拟loss下降
        print(f"Epoch {epoch+1}/5, Loss: {loss:.4f}")
    print("模型训练完成!")

def simulate_evaluation():
    """模拟评估阶段"""
    print("开始评估阶段...")
    for i in range(3):
        time.sleep(2)
        accuracy = random.uniform(0.7, 0.95)
        print(f"评估进度: {(i+1)*33}%, 准确率: {accuracy:.4f}")
    print("评估完成!")

def main():
    print("🚀 开始模拟实验...")
    print("实验总共预计需要约60秒")
    
    try:
        simulate_data_preparation()
        time.sleep(1)
        
        simulate_model_training()
        time.sleep(1)
        
        simulate_evaluation()
        
        print("✅ 实验成功完成!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⌨️  实验被用户中断")
        return 130
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 