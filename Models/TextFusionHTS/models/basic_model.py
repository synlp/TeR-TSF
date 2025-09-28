import torch.nn as nn
from utils.rev_in import RevIn
from models.PatchTST import Model as PatchTST

class BasicModel(nn.Module):
    
    def __init__(self, configs, device):
        super(BasicModel, self).__init__()
        self.args = configs
        self.device = device
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        
        tsmodel_dict = {
            'PatchTST': PatchTST
        }
        self.ts_model = tsmodel_dict[configs.ts_model](configs).float()
        
        if configs.revin:
            self.rev_in = RevIn(num_features=1)
            
    def forward(self, *args, **kwargs):
        pass