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


def get_args():
    parser = argparse.ArgumentParser(description='TFHTS Training')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--hist_len', type=int, default=36, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=6, help='Prediction sequence length')
    parser.add_argument('--text_type', type=str, default="original_text", help='')
    parser.add_argument('--data_name', type=str, default="Agriculture", help='')
    

    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feed forward network')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--patch_len', type=int, default=16, help='Length of patch')
    parser.add_argument('--stride', type=int, default=8, help='Stride of patch')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='', help='Path to save model')
    parser.add_argument('--time_label', type=int, default=0, help='label experiment with timestamp')
    parser.add_argument('--exp_time', type=str, default="001", help='experiment time label')
    parser.add_argument('--iter_idx', type=int, default=0, help='iteration index for reinforced data')
    parser.add_argument('--max_batch', type=int, default=-1, help='Maximum number of batches to process per epoch/test (-1 for all batches)')
    

    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()


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

        ts_data = [eval(ts) for ts in self.df['history_series']]
        pred_data = [eval(pred) for pred in self.df['horizon_series']]

        self.ts_data = (np.array(ts_data, dtype=np.float32) - self.mean) / self.std
        self.pred_data = (np.array(pred_data, dtype=np.float32) - self.mean) / self.std
        

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

        return {
            'ts': ts,
            'text_emb': text_emb,
            'pred': pred,
            'idx': idx
        }


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
        

        if max_batch > 0 and batch_count >= max_batch:
            break
    
    return total_loss / batch_count if batch_count > 0 else 0

def validate_model(model, val_loader, device, dataset, max_batch=-1):

    model.eval()
    total_mse = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            ts = batch['ts'].unsqueeze(-1).to(device)
            text_emb = batch['text_emb'].to(device)
            pred_true = batch['pred'].unsqueeze(-1).to(device) 
            

            pred = model(text_emb, ts)
            

            mse = torch.nn.functional.mse_loss(pred, pred_true)
            total_mse += mse.item()
            total_batches += 1
            

            if max_batch > 0 and total_batches >= max_batch:

                break
    

    avg_mse = total_mse / total_batches if total_batches > 0 else 0.0
    print(f"Val Batch Num {total_batches}, val_MSE = {avg_mse:.3f}")
    
    return avg_mse

def test(model, test_loader, device, dataset, max_batch=-1):

    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_samples = 0
    batch_count = 0
    

    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            ts = batch['ts'].unsqueeze(-1).to(device)
            text_emb = batch['text_emb'].to(device)
            pred_true = batch['pred'].unsqueeze(-1).to(device)
            

            pred = model(text_emb, ts)
            
            sample_mse = torch.mean((pred - pred_true) ** 2, dim=(1, 2))  # [batch_size]
            sample_mae = torch.mean(torch.abs(pred - pred_true), dim=(1, 2))  # [batch_size]
            

            total_squared_error += torch.sum(sample_mse).item()
            total_absolute_error += torch.sum(sample_mae).item()
            

            batch_size = pred.shape[0]
            total_samples += batch_size
            batch_count += 1
            

            if max_batch > 0 and batch_count >= max_batch:
                break

    avg_mse = total_squared_error / total_samples if total_samples > 0 else 0.0
    avg_mae = total_absolute_error / total_samples if total_samples > 0 else 0.0
    
    print(f"\n=== Test results ===")
    print(f"Total batch number: {batch_count}")
    print(f"Total sample number: {total_samples}")
    print(f"Test MSE: {avg_mse:.3f}")
    print(f"Test MAE: {avg_mae:.3f}")
    
    return avg_mse, avg_mae


def main():

    experiment_start_time = time.time()
    
    args = get_args()
    

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    

    if args.time_label > 0:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print(f"experiment start time: {current_time}")
        save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.hist_len}_{args.pred_len}_iter{args.iter_idx}_{current_time}")
    else:
        save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.hist_len}_{args.pred_len}_iter{args.iter_idx}_{args.exp_time}")
    os.makedirs(save_path, exist_ok=True)



    model_id = "/data2/user2/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    text_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    if args.iter_idx == 0 and args.text_type == "original_text":

        train_data_path = os.path.join("./processed_data", f"{args.data_name}_{args.hist_len}_{args.pred_len}_train.csv")
        val_data_path = os.path.join("./processed_data", f"{args.data_name}_{args.hist_len}_{args.pred_len}_val.csv")
        test_data_path = os.path.join("./processed_data", f"{args.data_name}_{args.hist_len}_{args.pred_len}_test.csv")
    else:

        train_data_path = os.path.join(args.data_dir, "train", f"iter{args.iter_idx}", f"{args.data_name}_{args.hist_len}_{args.pred_len}_gen0_{args.exp_time}.csv")

        val_data_path = os.path.join(args.data_dir, "val", f"iter{args.iter_idx}", f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv")
        test_data_path = os.path.join(args.data_dir, "test", f"iter{args.iter_idx}", f"{args.data_name}_{args.hist_len}_{args.pred_len}_{args.exp_time}.csv")

    
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


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    training_start_time = time.time()
    print(f"\n=== Training ===")

    best_loss = float('inf')
    best_val_mse = float('inf')
    patience_counter = 0 
    
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, args.device, args.max_batch)
        print(f"\nEpoch {epoch + 1}/{args.epochs}, Training Loss: {loss:.3f}")

        if loss < best_loss:
            best_loss = loss
        
        #
        if args.max_batch <= 0 and (epoch + 1) % args.val_interval == 0:
            val_mse = validate_model(model, val_loader, args.device, val_dataset, args.max_batch)
           
            

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0 
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss,
                    'val_mse': val_mse,
                }, os.path.join(save_path, f'{args.data_name}_{args.hist_len}_{args.pred_len}.pth'))
                print(f"saving best model (MSE: {val_mse:.3f}) ...")
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{args.patience}")
            

            if patience_counter >= args.patience:
                print(f"\n=== Early stopping triggered at epoch {epoch + 1} ===")
                print(f"Best validation MSE: {best_val_mse:.3f}")
                print(f"Patience: {args.patience}")
                break
    

    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    test_start_time = time.time()
    

    test_mse, test_mae = test(model, test_loader, args.device, test_dataset, args.max_batch)

    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    

    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    

    
    print(f"training loss: {best_loss:.6f}")
    print(f"val MSE: {best_val_mse:.6f}")
    print(f"test MSE: {test_mse:.6f}")
    print(f"test MAE: {test_mae:.6f}")
    
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
    
    record_result = RecordExpMetrics(os.path.join(args.save_dir, "tfhts_experiments.json"))
    
    result = {
        "best_train_loss": best_loss, 
        "best_val_mse": best_val_mse, 
        "test_mse": test_mse, 
        "test_mae": test_mae,
    }
    record_result.add_result(fixed_params, varying_params, result)

    print(test_mse + test_mae)

if __name__ == "__main__":
    main()