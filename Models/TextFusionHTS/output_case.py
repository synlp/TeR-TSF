import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

# 设置matplotlib全局参数以优化图片质量
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']  # 字体设置
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['figure.facecolor'] = 'white'  # 图片背景色
plt.rcParams['axes.facecolor'] = 'white'    # 坐标轴背景色
plt.rcParams['savefig.facecolor'] = 'white' # 保存时背景色
plt.rcParams['savefig.edgecolor'] = 'none'  # 保存时边框色
plt.rcParams['savefig.dpi'] = 300           # 默认保存DPI
plt.rcParams['savefig.bbox'] = 'tight'      # 默认裁剪方式

# 设置随机种子保证可重复性
np.random.seed(42)

# ===== 可调试的样式参数 =====
# 图例设置
LEGEND_ALPHA = 0.8          # 图例透明度 (0-1)
LEGEND_FONT_SIZE = 22       # 图例字体大小
LEGEND_SHADOW = False        # 图例阴影
LEGEND_FRAME_ON = True      # 图例边框

# 标题和标签字体大小
TITLE_FONT_SIZE = 20        # 标题字体大小
LABEL_FONT_SIZE = 22        # 坐标轴标签字体大小
TICK_FONT_SIZE = 22         # 坐标轴刻度字体大小

# 曲线样式
LINE_WIDTH_INPUT = 5        # 输入历史序列线宽
LINE_WIDTH_PRED = 5         # 预测序列线宽
LINE_WIDTH_TRUTH = 6        # 真实值线宽

# 图片尺寸和分辨率 - 优化用于PPT和论文
FIGURE_WIDTH = 16           # 图片宽度
FIGURE_HEIGHT = 10          # 图片高度
FIGURE_DPI = 300            # 图片分辨率 - 提高用于高质量输出

# 网格设置
GRID_ALPHA = 0.3            # 网格透明度
GRID_STYLE = True           # 是否显示网格

def get_args():
    parser = argparse.ArgumentParser(description='Generate case study visualization for specific sample')
    parser.add_argument('--idx', type=int, default=78, help='Sample index to visualize')
    parser.add_argument('--data_dir', type=str, default="/data2/user2/ter_tsf/visual", help='Data directory path')
    parser.add_argument('--data_name', type=str, default="SocialGood", help='Dataset name')
    parser.add_argument('--hist_len', type=int, default=36, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction sequence length')
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory path')
    return parser.parse_args()

def main():
    args = get_args()
    
    # 配置参数
    data_dir = args.data_dir
    data_name = args.data_name
    data_setting = f"{data_name}_{args.hist_len}_{args.pred_len}"
    
    model_folders = {
        "PatchTST": f"{data_setting}_iter0_001_use_text0",
        "TFHTS": f"{data_setting}_iter0_001_use_text1", 
        "TeR_TSF": f"{data_setting}_iter3_010_use_text1"
    }
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(data_dir, "imgs", data_setting)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载三个模型的数据
    model_data = {}
    for model_name, folder_name in model_folders.items():
        json_path = os.path.join(data_dir, folder_name, f"test_outputs_{data_name}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到文件: {json_path}")
        
        with open(json_path, 'r') as f:
            model_data[model_name] = json.load(f)
    
    # 获取样本数量
    sample_count = len(model_data["TeR_TSF"]["sample_indices"])
    
    # 检查idx是否有效
    if args.idx >= sample_count:
        raise ValueError(f"样本索引 {args.idx} 超出范围，总样本数为 {sample_count}")
    
    sample_idx = args.idx
    sample_id = model_data["TeR_TSF"]["sample_indices"][sample_idx]
    
    print(f"正在生成样本 {sample_id} (索引 {sample_idx}) 的对比图...")
    
    # 创建画布
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
    
    # 获取当前样本的数据
    input_ts = model_data["TeR_TSF"]["input_ts"][sample_idx]
    true_pred = model_data["TeR_TSF"]["true_predictions"][sample_idx]
    
    # 创建连续的时间轴，从0开始
    x_input = np.arange(len(input_ts))
    x_pred = np.arange(len(input_ts), len(input_ts) + len(true_pred))
    
    # 绘制输入历史时间序列
    line_input, = plt.plot(x_input, input_ts, 
                           color='#666666', linewidth=LINE_WIDTH_INPUT, linestyle='-', label='Input History')
    
    # 在历史窗口和预测窗口之间添加竖虚线分隔
    plt.axvline(x=len(input_ts)-1, color='#666666', linewidth=LINE_WIDTH_INPUT, linestyle='--', alpha=0.7)
    
    # 绘制三个模型的预测结果，连接到历史序列的最后一个点
    # 获取历史序列的最后一个值作为连接点
    last_input_value = input_ts[-1]
    
    # PatchTST预测
    patchtst_pred = model_data["PatchTST"]["model_predictions"][sample_idx]
    # 在历史序列末尾添加连接点，然后连接预测序列
    patchtst_connected = np.concatenate([[last_input_value], patchtst_pred])
    x_patchtst = np.arange(len(input_ts)-1, len(input_ts) + len(patchtst_pred))
    line_patchtst, = plt.plot(x_patchtst, patchtst_connected, 
                              color='#FFA245', linewidth=LINE_WIDTH_PRED, linestyle=':', label='PatchTST')
    
    # TFHTS预测
    tfhts_pred = model_data["TFHTS"]["model_predictions"][sample_idx]
    tfhts_connected = np.concatenate([[last_input_value], tfhts_pred])
    x_tfhts = np.arange(len(input_ts)-1, len(input_ts) + len(tfhts_pred))
    line_tfhts, = plt.plot(x_tfhts, tfhts_connected, 
                           color='#69B271', linewidth=LINE_WIDTH_PRED, linestyle='-.', label='TFHTS')
    
    # TeR-TSF预测
    ter_tsf_pred = model_data["TeR_TSF"]["model_predictions"][sample_idx]
    ter_tsf_connected = np.concatenate([[last_input_value], ter_tsf_pred])
    x_ter_tsf = np.arange(len(input_ts)-1, len(input_ts) + len(ter_tsf_pred))
    line_ter_tsf, = plt.plot(x_ter_tsf, ter_tsf_connected, 
                             color='#3C7EDA', linewidth=LINE_WIDTH_PRED, linestyle='-', label='TeR-TSF')
    
    # 绘制真实值，连接到历史序列的最后一个点
    true_connected = np.concatenate([[last_input_value], true_pred])
    x_true = np.arange(len(input_ts)-1, len(input_ts) + len(true_pred))
    line_truth, = plt.plot(x_true, true_connected, 
                           color='#E64A45', linewidth=LINE_WIDTH_TRUTH, linestyle='--', label='Ground Truth')
    
    # 添加图例
    legend = plt.legend(handles=[line_input, line_patchtst, line_tfhts, line_ter_tsf, line_truth],
               loc='upper left', 
               frameon=LEGEND_FRAME_ON, 
               shadow=LEGEND_SHADOW, 
               prop={'size': LEGEND_FONT_SIZE})
    
    # 设置图例透明度
    if hasattr(legend, 'get_frame'):
        legend.get_frame().set_alpha(LEGEND_ALPHA)
    
    # 添加标题和标签
    plt.title(f'Model Performance Comparison - Sample {sample_id}', fontsize=TITLE_FONT_SIZE, pad=20)
    plt.xlabel('Time Steps', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Normalized Values', fontsize=LABEL_FONT_SIZE)
    
    # 设置坐标轴刻度的字体大小
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    
    # 设置坐标轴
    if GRID_STYLE:
        plt.grid(True, alpha=GRID_ALPHA)
    
    # 调整整体布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{sample_id}.png")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"已生成样本 {sample_id} 的对比图: {output_path}")
    
    # 输出对应的文本信息
    print(f"\n=== 样本 {sample_id} 的文本信息 ===")
    
    # 输出TFHTS模型的文本
    if "input_texts" in model_data["TFHTS"]:
        tfhts_text = model_data["TFHTS"]["input_texts"][sample_idx]
        print(f"\nTFHTS模型输入文本:")
        print(f"{tfhts_text}")
    else:
        print(f"\nTFHTS模型: 无文本数据")
    
    # 输出TeR-TSF模型的文本
    if "input_texts" in model_data["TeR_TSF"]:
        ter_tsf_text = model_data["TeR_TSF"]["input_texts"][sample_idx]
        print(f"\nTeR-TSF模型输入文本:")
        print(f"{ter_tsf_text}")
    else:
        print(f"\nTeR-TSF模型: 无文本数据")
    
    # 输出时间序列数据统计信息
    print(f"\n=== 时间序列数据统计 ===")
    print(f"历史序列长度: {len(input_ts)}")
    print(f"预测序列长度: {len(true_pred)}")
    print(f"历史序列范围: [{np.min(input_ts):.3f}, {np.max(input_ts):.3f}]")
    print(f"真实预测范围: [{np.min(true_pred):.3f}, {np.max(true_pred):.3f}]")
    
    # 计算预测误差
    print(f"\n=== 预测误差分析 ===")
    patchtst_mse = np.mean((np.array(patchtst_pred) - np.array(true_pred)) ** 2)
    tfhts_mse = np.mean((np.array(tfhts_pred) - np.array(true_pred)) ** 2)
    ter_tsf_mse = np.mean((np.array(ter_tsf_pred) - np.array(true_pred)) ** 2)
    
    print(f"PatchTST MSE: {patchtst_mse:.6f}")
    print(f"TFHTS MSE: {tfhts_mse:.6f}")
    print(f"TeR-TSF MSE: {ter_tsf_mse:.6f}")
    
    print(f"\n所有信息已保存到: {output_dir}")

if __name__ == "__main__":
    main()
