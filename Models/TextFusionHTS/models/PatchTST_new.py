import torch
from torch import nn
# from ..layers.Transformer_EncDec import Encoder, EncoderLayer
# from ..layers.Attention_Family import FullAttention, AttentionLayer
# from ..layers.Embed import PatchEmbedding
from Models.TextFusionHTS.layers.Transformer_EncDec import Encoder, EncoderLayer
from Models.TextFusionHTS.layers.Attention_Family import FullAttention, AttentionLayer
from Models.TextFusionHTS.layers.Embed import PatchEmbedding

class Model(nn.Module):
    """
    PatchTST: A Time Series Transformer with Patch Input
    Paper: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, 
                seq_len: int,           # Input sequence length
                pred_len: int,          # Prediction sequence length
                d_model: int,           # Model dimension
                n_heads: int,           # Number of attention heads
                d_ff: int,             # Dimension of feed forward network
                e_layers: int,          # Number of encoder layers
                patch_len: int = 16,    # Length of each patch
                stride: int = 8,        # Stride between patches
                dropout: float = 0.1,   # Dropout rate
                activation: str = "gelu" # Activation function
                ):
        super().__init__()
        self.task_name = 'long_term_forecast'  # Match original model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Use the same padding as original model
        padding = stride
        
        # Patch embedding layer
        # Input: [batch_size, seq_len, n_vars] -> Output: [batch_size * n_vars, patch_num, d_model]
        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=dropout
        )

        # Encoder layers
        # Input: [batch_size * n_vars, patch_num, d_model] -> Output: [batch_size * n_vars, patch_num, d_model]
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model=d_model,
                        n_heads=n_heads
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
    def forecast(self, x_enc):
        # Input shape: [batch_size, seq_len, n_vars]
        # Permute to: [batch_size, n_vars, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        
        # Patch embedding
        # Output shape: [batch_size * n_vars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # Encoder processing
        # Output shape: [batch_size * n_vars, patch_num, d_model]
        enc_out, _ = self.encoder(enc_out)
        
        # Reshape output
        # Final shape: [batch_size, n_vars, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        
        return enc_out

    def forward(self, x_enc):
        """
        Args:
            x_enc: Input sequence, shape [batch_size, seq_len, n_vars]
        Returns:
            dec_out: Output tensor, shape [batch_size, n_vars, seq_len, d_model]
        """
        dec_out = self.forecast(x_enc)
        return dec_out 