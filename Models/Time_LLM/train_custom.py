import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import os
from torch.utils.data import DataLoader
import sys

project_root = "/home/user2/projects/TeR_TSF"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置环境变量
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# 导入自定义模块
from Models.Time_LLM.models.TimeLLM_Custom import Model as TimeLLM_Custom
from Models.Time_LLM.data_provider.custom_data_loader import Dataset_Custom_CSV


class EarlyStopping:
    """简化的早停类"""
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))


def vali(model, vali_loader, criterion, mae_metric, device):
    """验证函数"""
    total_loss = []
    total_mae_loss = []
    model.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(vali_loader):
            if len(batch_data) == 5:  # 自定义数据集返回5个元素
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts = batch_data
            else:  # 标准数据集返回4个元素
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                prompts = None
                
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input - 简化处理
            dec_inp = torch.zeros_like(batch_y).float().to(device)

            # 模型前向传播
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompts)
            
            # 计算损失
            loss = criterion(outputs, batch_y)
            mae_loss = mae_metric(outputs, batch_y)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()
    return total_loss, total_mae_loss


def main():
    parser = argparse.ArgumentParser(description='Time-LLM Custom Training')

    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--model_id', type=str, default='custom_model')
    parser.add_argument('--model_comment', type=str, default='train')
    parser.add_argument('--seed', type=int, default=2021)

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='./dataset')
    parser.add_argument('--data', type=str, default='Agriculture')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--checkpoints', type=str, default='/data2/user2/rl_tsf/Time-LLM/')

    parser.add_argument('--seq_len', type=int, default=36)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=6)

    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--llm_model', type=str, default='LLAMA')
    parser.add_argument('--llm_dim', type=int, default=4096)
    parser.add_argument('--llm_layers', type=int, default=6)

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    setting = f'{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_llm{args.llm_model}'
    
    print(f"Loading data from: {args.root_path}/{args.data_path}")
    
    # 数据加载
    train_data = Dataset_Custom_CSV(
        root_path=args.root_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        data_path=args.data_path,
        scale=True,
        percent=args.percent
    )
    
    vali_data = Dataset_Custom_CSV(
        root_path=args.root_path,
        flag='val',
        size=[args.seq_len, args.label_len, args.pred_len],
        data_path=args.data_path,
        scale=True
    )
    
    test_data = Dataset_Custom_CSV(
        root_path=args.root_path,
        flag='test',
        size=[args.seq_len, args.label_len, args.pred_len],
        data_path=args.data_path,
        scale=True
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(vali_data)}")
    print(f"Test samples: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, drop_last=True)
    vali_loader = DataLoader(vali_data, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, drop_last=True)

    # 初始化模型
    print("Initializing model...")
    model = TimeLLM_Custom(args).float()
    model.to(device)

    # 创建检查点目录
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    if not os.path.exists(path):
        os.makedirs(path)

    # 获取可训练参数
    trained_parameters = []
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trained_parameters.append(param)
            trainable_params += param.numel()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    train_steps = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    early_stopping = EarlyStopping(patience=args.patience)

    print(f"Starting training for {args.train_epochs} epochs...")
    
    # 训练循环
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.train_epochs}')
        
        for i, batch_data in enumerate(progress_bar):
            iter_count += 1
            model_optim.zero_grad()

            if len(batch_data) == 5:  # 自定义数据集
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts = batch_data
            else:  # 标准数据集
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                prompts = None

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input - 简化处理
            dec_inp = torch.zeros_like(batch_y).float().to(device)

            # 模型前向传播
            try:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompts)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                # 反向传播
                loss.backward()
                model_optim.step()
                scheduler.step()
                
                # 更新进度条
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue

        print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
        train_loss = np.average(train_loss)
        
        # 验证
        vali_loss, vali_mae_loss = vali(model, vali_loader, criterion, mae_metric, device)
        test_loss, test_mae_loss = vali(model, test_loader, criterion, mae_metric, device)
        
        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | "
              f"Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f} | "
              f"MAE Loss: {test_mae_loss:.7f}")

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Training completed!')
    print(f"Best model saved at: {path}/checkpoint.pth")

if __name__ == '__main__':
    main() 