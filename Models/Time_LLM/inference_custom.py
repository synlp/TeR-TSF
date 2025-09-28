import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 导入自定义模块
from models.TimeLLM_Custom import Model as TimeLLM_Custom
from data_provider.custom_data_loader import Dataset_Custom_CSV


def inference():
    parser = argparse.ArgumentParser(description='Time-LLM Custom Inference')

    # 基本配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--model_id', type=str, default='custom_model')
    parser.add_argument('--seed', type=int, default=2021)

    # 数据配置
    parser.add_argument('--data_path', type=str, required=True, help='path to CSV file')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--features', type=str, default='S', help='forecasting task')
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model checkpoint')

    # 预测任务配置
    parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=6, help='prediction sequence length')

    # 模型配置
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default=768, help='LLM model dimension')
    parser.add_argument('--llm_layers', type=int, default=6, help='number of LLM layers')

    # 推理配置
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using device: {device}")
    print(f"Loading data from: {args.data_path}")
    print(f"Model path: {args.model_path}")

    # 创建数据集（使用所有数据进行推理）
    inference_data = Dataset_Custom_CSV(
        root_path=args.root_path,
        flag='all',  # 使用所有数据进行推理
        size=[args.seq_len, args.label_len, args.pred_len],
        data_path=args.data_path,
        scale=True
    )

    inference_loader = DataLoader(
        inference_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        drop_last=False
    )

    print(f"Loaded {len(inference_data)} samples for inference")

    # 初始化模型
    model = TimeLLM_Custom(args).float()
    
    # 加载训练好的模型
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    model.to(device)
    model.eval()

    print("Starting inference...")
    
    predictions = []
    ground_truths = []
    sample_mse_list = []  # 存储每个样本的MSE
    
    with torch.no_grad():
        for batch_data in inference_loader:
            if len(batch_data) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts = batch_data
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                prompts = None
                
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            if args.label_len > 0:
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            else:
                dec_inp = torch.zeros_like(batch_y).float().to(device)

            # 模型推理
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompts)
            
            # 处理输出维度
            f_dim = -1 if args.features == 'MS' else 0
            if args.label_len > 0:
                batch_predictions = outputs[:, -args.pred_len:, f_dim:].cpu().numpy()
                batch_ground_truths = batch_y[:, -args.pred_len:, f_dim:].cpu().numpy()
            else:
                batch_predictions = outputs[:, :, f_dim:].cpu().numpy()
                batch_ground_truths = batch_y[:, :, f_dim:].cpu().numpy()
            
            # 计算每个样本的MSE
            for i in range(batch_predictions.shape[0]):
                sample_pred = batch_predictions[i].flatten()
                sample_true = batch_ground_truths[i].flatten()
                sample_mse = mean_squared_error(sample_true, sample_pred)
                sample_mse_list.append(sample_mse)
            
            predictions.extend(batch_predictions)
            ground_truths.extend(batch_ground_truths)

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    sample_mse_array = np.array(sample_mse_list)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Ground truth shape: {ground_truths.shape}")
    print(f"Sample MSE array shape: {sample_mse_array.shape}")

    # 整体评估结果（用于显示）
    overall_mse = mean_squared_error(ground_truths.flatten(), predictions.flatten())
    overall_mae = mean_absolute_error(ground_truths.flatten(), predictions.flatten())
    
    print(f"\nOverall Evaluation Results:")
    print(f"Overall MSE: {overall_mse:.6f}")
    print(f"Overall MAE: {overall_mae:.6f}")
    print(f"Overall RMSE: {np.sqrt(overall_mse):.6f}")
    
    print(f"\nSample-wise MSE Statistics:")
    print(f"Mean sample MSE: {sample_mse_array.mean():.6f}")
    print(f"Std sample MSE: {sample_mse_array.std():.6f}")
    print(f"Min sample MSE: {sample_mse_array.min():.6f}")
    print(f"Max sample MSE: {sample_mse_array.max():.6f}")

    # Load original CSV and add reward1 column
    original_csv_path = os.path.join(args.root_path, args.data_path)
    df = pd.read_csv(original_csv_path)
    
    # 确保样本数量匹配
    if len(df) != len(sample_mse_array):
        raise ValueError(f"Sample count mismatch: CSV has {len(df)} rows, but got {len(sample_mse_array)} MSE values")
    
    # Add sample-specific MSE negative value as reward1
    df['reward1'] = -sample_mse_array
    
    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with model parameters
    base_name = os.path.splitext(args.data_path)[0]
    output_filename = f"{base_name}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_llm{args.llm_model}.csv"
    output_path = os.path.join(results_dir, output_filename)
    
    # Save to results directory
    df.to_csv(output_path, index=False)
    
    print(f"\nResults saved:")
    print(f"Updated CSV with reward1 column: {output_path}")
    print(f"Reward1 values (negative sample MSE) - Mean: {-sample_mse_array.mean():.6f}, Range: [{-sample_mse_array.max():.6f}, {-sample_mse_array.min():.6f}]")


if __name__ == '__main__':
    inference() 