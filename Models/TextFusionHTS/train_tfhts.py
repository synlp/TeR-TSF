import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import datetime
import random
import time
# import swanlab
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Add the project root directory to Python path
sys.path.insert(0, "/home/user2/projects/TeR_TSF")

from Models.TextFusionHTS.models.PatchTST_new import Model as PatchTST
from Models.TextFusionHTS.models.TFHTS_new import Model as TFHTS
from utils.tools import RecordExpMetrics

stat_dict = {
    "Agriculture": {
        "mean": 144.59045661,
        "std": 23.02460005,
    },
    "Climate": {
        "mean": 56.13323596,
        "std": 10.09066259,
    },
    "Economy": {
        "mean": -32983.5974359,
        "std": 23032.45317885,
    },
    "Energy": {
        "mean": 2.12598414,
        "std": 0.97018236,
    },
    "Environment": {
        "mean": 87.56969155,
        "std": 42.80683806,
    },
    "Health_US": {
        "mean": 1.57049671,
        "std": 1.20796971
    },
    "SocialGood": {
        "mean": 5.63603744,
        "std": 1.63104511
    },
    "Traffic": {
        "mean": 171844.25274725,
        "std": 51870.70593244
    },
    "ETTh1": {
        "mean": 16.29471487,
        "std": 8.34847203
    },
    "exchange_rate":{
        "mean": 0.60482487,
        "std": 0.0952995
    },
    "weather": {
        "mean": 59.6538856,
        "std": 16.95276253
    },
    "Heart_Rate": {
        "mean": 160.59121974,
        "std": 9.57681761
    },
    "MTBench_weather": {
        "mean": 15.826329,
        "std": 10.204743,
        "freq": 'h'
    },
    "MTBench_finance": {
        "mean": 0.0,
        "std": 1.0,
        "freq": 'h'
    }
}


# ===== 参数设置 =====
def get_args():
    parser = argparse.ArgumentParser(description='TFHTS Training')
    # 数据相关
    parser.add_argument('--data_dir', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--hist_len', type=int, default=36, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=6, help='Prediction sequence length')
    parser.add_argument('--text_type', type=str, default="original_text", help='')
    parser.add_argument('--data_name', type=str, default="Agriculture", help='')
    
    # 模型相关
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feed forward network')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--patch_len', type=int, default=16, help='Length of patch')
    parser.add_argument('--stride', type=int, default=8, help='Stride of patch')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='', help='Path to save model')
    parser.add_argument('--time_label', type=int, default=0, help='label experiment with timestamp')
    parser.add_argument('--exp_time', type=str, default="001", help='experiment time label')
    parser.add_argument('--iter_idx', type=int, default=0, help='iteration index for reinforced data')
    parser.add_argument('--max_batch', type=int, default=-1, help='Maximum number of batches to process per epoch/test (-1 for all batches)')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

# ===== 数据集类 =====
class TFHTSDataset(Dataset):
    def __init__(self, data_path, hist_len, pred_len, text_type, tokenizer, text_model, data_name, device):
        self.df = pd.read_csv(data_path)
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.device = device
        self.data_name = data_name
        self.text_type = text_type
        self.mean, self.std = stat_dict[self.data_name]["mean"], stat_dict[self.data_name]["std"]
        # 预处理时间序列数据
        ts_data = [eval(ts) for ts in self.df['history_series']]
        pred_data = [eval(pred) for pred in self.df['horizon_series']]
        # import pdb; pdb.set_trace()
        self.ts_data = (np.array(ts_data, dtype=np.float32) - self.mean) / self.std
        self.pred_data = (np.array(pred_data, dtype=np.float32) - self.mean) / self.std
        
        # 预处理文本数据并提取嵌入
        print("Extracting text embeddings...")
        self.text_embeddings = self._extract_text_embeddings()
    
    def _extract_text_embeddings(self):
        embeddings = []
        
        texts = self.df[self.text_type].astype(str).tolist()
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Processing texts"):
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                      truncation=True, max_length=512).to(self.device)
                outputs = self.text_model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                embedding = hidden.mean(dim=1).cpu().numpy()[0]
                embeddings.append(embedding)
                
        return np.array(embeddings)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ts = self.ts_data[idx]
        pred = self.pred_data[idx]
        text_emb = self.text_embeddings[idx].astype(np.float32)
        # import pdb; pdb.set_trace()
        return {
            'ts': ts,
            'text_emb': text_emb,
            'pred': pred,
            'idx': idx
        }

# ===== 训练函数 =====
def train_epoch(model, train_loader, optimizer, criterion, device, max_batch=-1):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        ts = batch['ts'].unsqueeze(-1).to(device)
        text_emb = batch['text_emb'].to(device)
        pred_true = batch['pred'].unsqueeze(-1).to(device)
        
        pred = model(text_emb, ts)
        loss = criterion(pred, pred_true)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # 如果设置了max_batch限制且达到限制，提前退出
        if max_batch > 0 and batch_count >= max_batch:
            print(f"达到最大批次限制 {max_batch}，提前结束训练")
            break
    
    return total_loss / batch_count if batch_count > 0 else 0

def validate_model(model, val_loader, device, dataset, max_batch=-1):
    """验证模型在整个验证数据集上的预测效果"""
    model.eval()
    total_mse = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            ts = batch['ts'].unsqueeze(-1).to(device)
            text_emb = batch['text_emb'].to(device)
            pred_true = batch['pred'].unsqueeze(-1).to(device)  # 添加维度以匹配预测输出
            
            # 获取预测
            pred = model(text_emb, ts)
            
            # 直接计算batch上的MSE（不反归一化）
            mse = torch.nn.functional.mse_loss(pred, pred_true)
            total_mse += mse.item()
            total_batches += 1
            
            # 如果设置了max_batch限制且达到限制，提前退出
            if max_batch > 0 and total_batches >= max_batch:
                print(f"达到最大批次限制 {max_batch}，提前结束验证")
                break
    
    # 计算平均MSE
    avg_mse = total_mse / total_batches if total_batches > 0 else 0.0
    print(f"Val Batch Num {total_batches}, val_MSE = {avg_mse:.3f}")
    
    return avg_mse

def test(model, test_loader, device, dataset, max_batch=-1):
    """测试模型在整个测试数据集上的预测效果"""
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_samples = 0
    batch_count = 0
    
    print("\n=== 开始测试评估 ===")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试中")):
            ts = batch['ts'].unsqueeze(-1).to(device)
            text_emb = batch['text_emb'].to(device)
            pred_true = batch['pred'].unsqueeze(-1).to(device)
            
            # 获取预测
            pred = model(text_emb, ts)
            
            # 计算每个样本的MSE和MAE（在时间序列维度上平均）
            # pred shape: [batch_size, pred_len, 1]
            # pred_true shape: [batch_size, pred_len, 1]
            sample_mse = torch.mean((pred - pred_true) ** 2, dim=(1, 2))  # [batch_size]
            sample_mae = torch.mean(torch.abs(pred - pred_true), dim=(1, 2))  # [batch_size]
            
            # 累加每个样本的误差
            total_squared_error += torch.sum(sample_mse).item()
            total_absolute_error += torch.sum(sample_mae).item()
            
            # 计算当前batch的样本数
            batch_size = pred.shape[0]
            total_samples += batch_size
            batch_count += 1
            
            # 如果设置了max_batch限制且达到限制，提前退出
            if max_batch > 0 and batch_count >= max_batch:
                print(f"达到最大批次限制 {max_batch}，提前结束测试")
                break
            
    # 计算最终的MSE和MAE（样本平均）
    avg_mse = total_squared_error / total_samples if total_samples > 0 else 0.0
    avg_mae = total_absolute_error / total_samples if total_samples > 0 else 0.0
    
    print(f"\n=== Test results ===")
    print(f"Total batch number: {batch_count}")
    print(f"Total sample number: {total_samples}")
    print(f"Test MSE: {avg_mse:.3f}")
    print(f"Test MAE: {avg_mae:.3f}")
    
    return avg_mse, avg_mae

# ===== 主函数 =====
def main():
    # 记录实验开始时间
    experiment_start_time = time.time()
    
    args = get_args()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # 创建保存目录
    # 获取当前日期时间字符串
    if args.time_label > 0:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print(f"experiment start time: {current_time}")
        save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.hist_len}_{args.pred_len}_iter{args.iter_idx}_{current_time}")
    else:
        save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.hist_len}_{args.pred_len}_iter{args.iter_idx}_{args.exp_time}")
    os.makedirs(save_path, exist_ok=True)

    # swanlab.login(api_key="L212bDNNAv5lbvTBzAIF7")
    # if args.time_label > 0:
    #     experiment_name = f"{args.data_name}_{args.hist_len}_{args.pred_len}_iter{args.iter_idx}_{current_time}"
    # else:
    #     experiment_name = f"{args.data_name}_{args.hist_len}_{args.pred_len}_iter{args.iter_idx}_{args.exp_time}"
    # swanlab.init(
    #             mode="disabled",
    #             project="TeR-TSF-Exp",
    #             experiment_name=experiment_name,
    #             config=vars(args)
    #             )
    
    # 设置文本模型
    model_id = "/data2/user2/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    text_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 根据iter_idx和text_type确定数据路径
    if args.iter_idx == 0 and args.text_type == "original_text":
        # 第0轮使用original_text：直接使用原始数据
        train_data_path = os.path.join("/data2/user2/ter_tsf/processed_data", f"{args.data_name}_{args.hist_len}_{args.pred_len}_train.csv")
        val_data_path = os.path.join("/data2/user2/ter_tsf/processed_data", f"{args.data_name}_{args.hist_len}_{args.pred_len}_val.csv")
        test_data_path = os.path.join("/data2/user2/ter_tsf/processed_data", f"{args.data_name}_{args.hist_len}_{args.pred_len}_test.csv")
    else:
        # 其他情况使用reinforced_data目录结构
        # 训练集：使用当前轮次的数据（使用gen0作为训练数据）
        train_data_path = os.path.join(args.data_dir, "train", f"iter{args.iter_idx}", f"{args.data_name}_{args.hist_len}_{args.pred_len}_gen0_{args.exp_time}.csv")
        # 验证集和测试集：使用untrained基线数据
        # val_data_path = os.path.join(args.data_dir, "val", "untrained", f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv")
        # test_data_path = os.path.join(args.data_dir, "test", "untrained", f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv")
        val_data_path = os.path.join(args.data_dir, "val", f"iter{args.iter_idx}", f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv")
        test_data_path = os.path.join(args.data_dir, "test", f"iter{args.iter_idx}", f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv")
    # 打印数据路径信息
    print(f"训练数据路径: {train_data_path}")
    print(f"验证数据路径: {val_data_path}")
    print(f"测试数据路径: {test_data_path}")
    print(f"文本类型: {args.text_type}")
    print(f"迭代索引: {args.iter_idx}")
    if args.max_batch > 0:
        print(f"最大批次限制: {args.max_batch} (快速测试模式)")
    else:
        print(f"最大批次限制: 无限制 (完整训练模式)")
    
    # 创建数据集和数据加载器
    train_dataset = TFHTSDataset(train_data_path, args.hist_len, args.pred_len, args.text_type,
                          tokenizer, text_model, args.data_name, args.device)
    val_dataset = TFHTSDataset(val_data_path, args.hist_len, args.pred_len, args.text_type,
                          tokenizer, text_model, args.data_name, args.device)
    test_dataset = TFHTSDataset(test_data_path, args.hist_len, args.pred_len, args.text_type,
                          tokenizer, text_model, args.data_name, args.device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, drop_last=True, num_workers=10)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, drop_last=False, num_workers=10)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, drop_last=False, num_workers=10)
    
    # 创建模型
    model = TFHTS(
        seq_len=args.hist_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        patch_len=args.patch_len,
        stride=args.stride,
        d_txt=text_model.config.hidden_size,
        dropout=args.dropout,
        activation="gelu",
        device=args.device
    ).to(args.device)

    # model.load_state_dict(torch.load("./checkpoints/agriculture_36_6.pth"))

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    # 记录训练开始时间
    training_start_time = time.time()
    print(f"\n=== 开始训练 ===")
    
    # 训练循环
    best_loss = float('inf')
    best_val_mse = float('inf')
    patience_counter = 0  # 早停计数器
    
    for epoch in range(args.epochs):
        # 训练阶段
        loss = train_epoch(model, train_loader, optimizer, criterion, args.device, args.max_batch)
        print(f"\nEpoch {epoch + 1}/{args.epochs}, Training Loss: {loss:.3f}")
        # swanlab.log({"train/train_loss": loss}, step=epoch)
        if loss < best_loss:
            best_loss = loss # 记录最佳训练损失
        
        # 每个val_interval个epoch验证一次 (快速测试模式下跳过验证)
        if args.max_batch <= 0 and (epoch + 1) % args.val_interval == 0:
            val_mse = validate_model(model, val_loader, args.device, val_dataset, args.max_batch)
            # swanlab.log({"val/mse": val_mse}, step=epoch)
            
            # 保存最佳验证MSE模型
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0  # 重置早停计数器
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss,
                    'val_mse': val_mse,
                }, os.path.join(save_path, f'{args.data_name}_{args.hist_len}_{args.pred_len}.pth'))
                print(f"saving best model (MSE: {val_mse:.3f}) ...")
            else:
                patience_counter += 1  # 增加早停计数器
                print(f"Early stopping counter: {patience_counter}/{args.patience}")
            
            # 早停检查
            if patience_counter >= args.patience:
                print(f"\n=== Early stopping triggered at epoch {epoch + 1} ===")
                print(f"Best validation MSE: {best_val_mse:.3f}")
                print(f"Patience: {args.patience}")
                break
    
    # 记录训练结束时间
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    print(f"\n=== 训练完成 ===")
    print(f"训练总时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
    
    # 快速测试模式：跳过测试阶段，只输出训练时间
    if args.max_batch > 0:
        print(f"\n=== 快速测试模式完成 ===")
        print(f"处理批次数: {args.max_batch}")
        print(f"训练时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
        print(f"平均每批次时间: {training_time/args.max_batch:.4f} 秒")
        print(f"最佳训练损失: {best_loss:.6f}")
        print("\n注意: 快速测试模式下跳过了验证和测试阶段")
        return  # 提前结束，不进行测试和结果记录
    
    # 完整模式：进行测试评估
    # 记录测试开始时间
    test_start_time = time.time()
    
    # 训练结束后进行最终测试评估
    print("\n=== 最终测试评估 ===")
    test_mse, test_mae = test(model, test_loader, args.device, test_dataset, args.max_batch)
    
    # 记录测试结束时间
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    
    # 计算总实验时间
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    # 输出时间统计信息
    print(f"\n=== 时间统计 ===")
    print(f"训练时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
    print(f"测试时间: {test_time:.2f} 秒 ({test_time/60:.2f} 分钟)")
    print(f"总实验时间: {total_experiment_time:.2f} 秒 ({total_experiment_time/60:.2f} 分钟)")
    
    # 输出最终结果汇总
    print(f"\n=== 最终结果汇总 ===")
    print(f"最佳训练损失: {best_loss:.6f}")
    print(f"最佳验证MSE: {best_val_mse:.6f}")
    print(f"测试MSE: {test_mse:.6f}")
    print(f"测试MAE: {test_mae:.6f}")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"测试时间: {test_time:.2f}秒")
    print(f"总时间: {total_experiment_time:.2f}秒")
    
    # 保存最终测试结果
    fixed_params = {
        'Data': args.data_name, 'hist_len': args.hist_len, 'pred_len': args.pred_len
    }
    if args.time_label:
        varying_params = {
            'text': args.text_type, 'lr': args.lr, 'epochs': args.epochs, 'time': current_time, 'iter_idx': args.iter_idx
        }
    else:
        varying_params = {
            'text': args.text_type, 'lr': args.lr, 'epochs': args.epochs, 'time': args.exp_time, 'iter_idx': args.iter_idx
        }
    
    # 记录实验结果（包含时间信息）
    record_result = RecordExpMetrics(os.path.join(args.save_dir, "tfhts_experiments.json"))
    result_label = f"hist{args.hist_len}_pred{args.pred_len}_iter{args.iter_idx}_{args.text_type}"
    result = {
        "best_train_loss": best_loss, 
        "best_val_mse": best_val_mse, 
        "test_mse": test_mse, 
        "test_mae": test_mae,
        "training_time_seconds": training_time,
        "test_time_seconds": test_time,
        "total_experiment_time_seconds": total_experiment_time,
        "training_time_minutes": training_time/60,
        "test_time_minutes": test_time/60,
        "total_experiment_time_minutes": total_experiment_time/60
    }
    record_result.add_result(fixed_params, varying_params, result)

    print(test_mse + test_mae)

    # swanlab.log({"test/mse": test_mse, "test/mae": test_mae}, step=99999)
    # swanlab.finish()

if __name__ == "__main__":
    main()