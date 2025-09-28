import torch
import torch.nn as nn
# from ..models.PatchTST_new import Model as PatchTST
# from ..layers.Attention_Family import CrossAttention
from Models.TextFusionHTS.models.PatchTST_new import Model as PatchTST
from Models.TextFusionHTS.layers.Attention_Family import CrossAttention

class Model(nn.Module):
    def __init__(self,
                seq_len: int,           # Input sequence length
                pred_len: int,          # Prediction sequence length
                d_model: int,           # Model dimension
                n_heads: int,           # Number of attention heads
                d_ff: int,              # Dimension of feed forward network
                e_layers: int,          # Number of encoder layers
                patch_len: int = 16,    # Length of each patch
                stride: int = 8,        # Stride between patches
                d_txt: int = 4096,      # Input text embedding dimension
                projection_dim: int = 768, # Projection dimension for text
                dropout: float = 0.1,   # Dropout rate
                activation: str = "gelu", # Activation function
                device: str = "cuda"    # Device to run the model on
                ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device
        self.d_txt = d_txt
        
        # Check if text modality is enabled
        self.use_text = d_txt > 0
        
        # Time series model (PatchTST)
        # Input: [batch_size, seq_len, n_vars] -> Output: [batch_size, n_vars, patch_num, d_model]
        self.ts_model = PatchTST(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
            activation=activation
        )
        
        if self.use_text:
            # Text projection layers (only when text modality is enabled)
            # Input: [batch_size, d_txt] -> Output: [batch_size, projection_dim]
            self.txt_fc = nn.Linear(d_txt, projection_dim)
            
            # Text projection MLP
            # Input: [batch_size, projection_dim] -> Output: [batch_size, projection_dim]
            self.projection_mlp = nn.Sequential(
                nn.Linear(projection_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )
            
            # Cross attention for fusion
            # Input: ([batch_size, n_vars, patch_num, d_model], [batch_size, 1, projection_dim])
            # Output: [batch_size, 1, d_model]
            self.crossatn = CrossAttention(d_model, projection_dim)
            
            # Final prediction layer for multimodal mode
            # Input: [batch_size, 1, d_model] -> Output: [batch_size, pred_len, 1]
            self.linear = nn.Linear(d_model, pred_len)
        else:
            # Single-modal mode: direct prediction from time series features
            # Input: [batch_size, n_vars, patch_num, d_model] -> Output: [batch_size, pred_len, n_vars]
            self.linear = nn.Linear(d_model, pred_len)

    def forward(self, txt_enc=None, x_enc=None, means=None, stdev=None):
        """
        Args:
            txt_enc: Text embeddings [batch_size, d_txt] (optional when use_text=False)
            x_enc: Time series input [batch_size, seq_len, n_vars]
            means: Optional normalization mean [batch_size, 1, n_vars]
            stdev: Optional normalization std [batch_size, 1, n_vars]
        Returns:
            outputs: Predictions [batch_size, pred_len, n_vars]
        """
        # Handle single-modal mode (only time series)
        if not self.use_text:
            # Single-modal mode: only use time series data
            if x_enc is None:
                raise ValueError("x_enc is required for single-modal mode")
            
            # Normalization [batch_size, seq_len, n_vars]
            if means is None or stdev is None:
                means = x_enc.mean(1, keepdim=True).detach()
                x_enc = x_enc - means
                stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                x_enc = x_enc / stdev

            # Time series encoding
            # Input: [batch_size, seq_len, n_vars] 
            # Output: [batch_size, n_vars, patch_num, d_model]
            ts_emb = self.ts_model(x_enc)
            
            # Direct prediction from time series features
            # Input: [batch_size, n_vars, patch_num, d_model] -> Output: [batch_size, n_vars, pred_len]
            batch_size, n_vars, patch_num, d_model = ts_emb.shape
            ts_emb_flat = ts_emb.view(batch_size * n_vars, patch_num, d_model)
            
            # Global average pooling over patch dimension
            ts_emb_pooled = ts_emb_flat.mean(dim=1)  # [batch_size * n_vars, d_model]
            
            # Reshape back to [batch_size, n_vars, d_model]
            ts_emb_pooled = ts_emb_pooled.view(batch_size, n_vars, d_model)
            
            # Final prediction
            # Input: [batch_size, n_vars, d_model] -> Output: [batch_size, n_vars, pred_len]
            dec_out = self.linear(ts_emb_pooled)  # [batch_size, n_vars, pred_len]
            
            # Transpose to match expected output format [batch_size, pred_len, n_vars]
            dec_out = dec_out.permute(0, 2, 1)
            
            # De-Normalization
            outputs = dec_out * stdev + means
            
            return outputs
        
        else:
            # Multimodal mode: use both text and time series
            if txt_enc is None or x_enc is None:
                raise ValueError("Both txt_enc and x_enc are required for multimodal mode")
            
            # Normalization [batch_size, seq_len, n_vars]
            if means is None or stdev is None:
                means = x_enc.mean(1, keepdim=True).detach()
                x_enc = x_enc - means
                stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                x_enc = x_enc / stdev

            # Time series encoding
            # Input: [batch_size, seq_len, n_vars] 
            # Output: [batch_size, n_vars, patch_num, d_model]
            ts_emb = self.ts_model(x_enc)
            
            # Text encoding
            # Input: [batch_size, d_txt] -> Output: [batch_size, projection_dim]
            txt_emb = self.txt_fc(txt_enc)
            txt_emb = self.projection_mlp(txt_emb)
            # Reshape for cross attention: [batch_size, 1, projection_dim]
            txt_emb = txt_emb.unsqueeze(1)
            
            # Cross attention fusion
            # Input: ts_emb [batch_size, n_vars, patch_num, d_model], txt_emb [batch_size, 1, projection_dim]
            # Output: [batch_size, 1, d_model]
            enc_out, _ = self.crossatn(ts_emb, txt_emb)
            
            # Final prediction
            # Input: [batch_size, 1, d_model] -> Output: [batch_size, pred_len, 1]
            dec_out = self.linear(enc_out)
            dec_out = dec_out.permute(0, 2, 1)
            
            # De-Normalization
            outputs = dec_out * stdev + means
            
            return outputs