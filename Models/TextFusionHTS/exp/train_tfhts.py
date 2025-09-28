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

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the project root directory to Python path
PROJECT_ROOT = '/media/ubuntu/data/collaborations/tsf/TeR-TSF/transformer-based-tsf'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models.PatchTST_new import Model as PatchTST
    from models.TFHTS_new import Model as TFHTS
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

def get_args():
    parser = argparse.ArgumentParser(description='TFHTS Training')
  
    parser.add_argument('--data_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction sequence length')
    
 
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feed forward network')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--patch_len', type=int, default=16, help='Length of patch')
    parser.add_argument('--stride', type=int, default=8, help='Stride of patch')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Path to save model')
    
    return parser.parse_args()

class TFHTSDataset(Dataset):
    def __init__(self, data_path, seq_len, pred_len, tokenizer, text_model, device):
        self.df = pd.read_csv(data_path)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.device = device
        
        self.ts_data = [eval(ts) for ts in self.df['ts']]
        self.pred_data = [eval(pred) for pred in self.df['pred']]
        
        print("Normalizing time series data...")
        self.scalers = self._normalize_data()
        
        print("Extracting text embeddings...")
        self.text_embeddings = self._extract_text_embeddings()
    
    def _normalize_data(self):
        scalers = []
        for i in range(len(self.ts_data)):
            ts_data = np.array(self.ts_data[i])
            pred_data = np.array(self.pred_data[i])
            mean = np.mean(ts_data)
            std = np.std(ts_data)
            if std == 0:
                std = 1
            
            self.ts_data[i] = (ts_data - mean) / std
            self.pred_data[i] = (pred_data - mean) / std
            
            scalers.append((mean, std))
        return scalers
    
    def denormalize(self, normalized_data, idx):
        mean, std = self.scalers[idx]
        return normalized_data * std + mean
    
    def _extract_text_embeddings(self):
        embeddings = []
        texts = self.df['txt'].astype(str).tolist()
        
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
        ts = np.array(self.ts_data[idx], dtype=np.float32)
        pred = np.array(self.pred_data[idx], dtype=np.float32)
        text_emb = self.text_embeddings[idx].astype(np.float32)
        
        return {
            'ts': ts,
            'text_emb': text_emb,
            'pred': pred,
            'idx': idx
        }

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
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
    
    return total_loss / len(train_loader)

def validate_predictions(model, dataset, device, batch_idx=0):
    model.eval()
    with torch.no_grad():
        batch = dataset[batch_idx]
        ts = torch.tensor(batch['ts']).unsqueeze(0).unsqueeze(-1).to(device)
        text_emb = torch.tensor(batch['text_emb']).unsqueeze(0).to(device)
        pred_true = batch['pred']
        
        pred = model(text_emb, ts).cpu().numpy()[0, :, 0]

        pred_denorm = dataset.denormalize(pred, batch_idx)
        pred_true_denorm = dataset.denormalize(pred_true, batch_idx)
    
        mse = np.mean((pred_denorm - pred_true_denorm) ** 2)
        print(f"MSE: {mse:.4f}")
        
        return mse

def main():
    args = get_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    text_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    dataset = TFHTSDataset(args.data_path, args.seq_len, args.pred_len, 
                          tokenizer, text_model, args.device)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=0) 
    

    model = TFHTS(
        seq_len=args.seq_len,
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
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        print(f"\nEpoch {epoch + 1}/{args.epochs}, Loss: {loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            total_mse = 0
            for i in range(len(dataset)):
                mse = validate_predictions(model, dataset, args.device, i)
                total_mse += mse
            print(f"\Mean MSE: {total_mse/len(dataset):.4f}")
        
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(args.save_path, 'best_model.pth'))
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(args.save_path, f'checkpoint_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    main() 