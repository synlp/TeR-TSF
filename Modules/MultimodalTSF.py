import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from Models.ChatTime.model.model import ChatTime
from Models.MCD_TSF.main_model import CSDI_Forecasting
from Models.TextFusionHTS.models.TFHTS_new import Model as TFHTS
from Models.Time_LLM.models.TimeLLM_Custom import Model as TimeLLM
from utils.tools import Config
import os
import yaml
import json

chattime_checkpoints = "/home/suchen/ChatTime/ChatTime-1-7B-Chat"

mcd_config_dir = "./Models/MCD_TSF/config"
mcd_checkpoints = "/data2/user2/ter_tsf/pretrained_models/MCD_TSF"

hts_checkpoints = "/data2/user2/ter_tsf"

time_llm_checkpoints = "/data2/user2/ter_tsf/Time-LLM/"

def ChatTimeInit(hist_len, pred_len, domain, device, freq):
    model_path = chattime_checkpoints
    model = ChatTime(hist_len=hist_len, pred_len=pred_len, model_path=model_path)
    return model

def MCD_TSFInit(hist_len, pred_len, domain, device, freq):
    key = f"{domain}_{hist_len}_{pred_len}"
    mcd_path = mcd_checkpoints + '/' + key
    model_path = next(
    os.path.join(mcd_path, f) 
    for f in os.listdir(mcd_path) 
    if f.startswith("model") and os.path.isfile(os.path.join(mcd_path, f))
    )
    config_path = os.path.join(mcd_config_dir, key.lower()+".yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    guide_w = eval(model_path.split("_")[-1][:-4])
    model = CSDI_Forecasting(config, device, 1, window_lens=[hist_len, pred_len], freq=freq, guide_w=guide_w).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def TextFusionHTSInit(hist_len, pred_len, domain, device, freq, llm_type, iter_=0, exp_time="001"):
    key = f"{domain}_{hist_len}_{pred_len}"
    if iter_ == 0:
        model_path = os.path.join(hts_checkpoints, "TextFusionHTS", key+".pth")
    else:
        model_path = os.path.join(hts_checkpoints, llm_type, "tfhts", domain, "saved_models", f"{domain}_{hist_len}_{pred_len}_iter{iter_-1}_{exp_time}", key+".pth")
    if domain == "weather" or domain == "Heart_Rate":
        patch_len = 4
        stride = 2
    else:
        patch_len = 16
        stride = 8
    ts_model = TFHTS(hist_len, pred_len, d_model=128, n_heads=16, d_ff=256, e_layers=3, patch_len=patch_len, stride=stride, dropout=0.1, d_txt=4096, activation="gelu").to(device)
    ts_model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return ts_model

def TimeLLMInit(hist_len, pred_len, domain, device, freq):
    if domain == "weather" or domain == "Heart_Rate":
        patch_len = 4
        stride = 2
    else:
        patch_len = 16
        stride = 8
    config = Config(seq_len=hist_len, pred_len=pred_len, patch_len=patch_len, stride=stride)
    model = TimeLLM(config).to(device).float()
    model_path = os.path.join(time_llm_checkpoints, f"{domain}_sl{hist_len}_pl{pred_len}_dm16_llmLLAMA-train", "checkpoint.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

bm_model_init_dict = {
    "chattime": ChatTimeInit,
    "mcd-tsf": MCD_TSFInit,
    "textfusionhts": TextFusionHTSInit,
    "time-llm": TimeLLMInit
}

class BaseMultimodalTSFModel():
    def __init__(self, hist_len, pred_len, domain, bm_name="chattime", device="cuda:0", freq='d', llm_type="qwen3-1.7b", iter_=0, exp_time="001"):
        self.bm_name = bm_name
        self.bm_model = bm_model_init_dict[bm_name.lower()](hist_len, pred_len, domain, device, freq=freq, llm_type=llm_type, iter_=iter_, exp_time=exp_time)
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.device = device
    
    def chattime_predict_(self, hist_series, reinforced_text):
        with torch.no_grad():
            assert self.bm_name == "chattime"
            batch_size = len(hist_series)
            pred_results = []
            for idx in range(batch_size):
                pred = self.bm_model.predict(hist_series[idx], reinforced_text[idx])
                pred_results.append(pred)
        return pred_results
    
    def mcd_tsf_predict_(self, batch_data, nsample):
        with torch.no_grad():
            output = self.bm_model.evaluate(batch_data, nsample)
            samples, c_target, eval_points, observed_points, observed_time = output
            pred_out = samples.mean(dim=1).squeeze()
        return pred_out[:, self.hist_len:]
    
    def tfhts_predict_(self, text_emb, ts):
        with torch.no_grad():
            ts = ts.unsqueeze(-1).to(self.device)
            text_emb = text_emb.to(self.device)
            pred = self.bm_model(text_emb, ts)
        return pred
    
    def time_llm_predict_(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, prompts):
        with torch.no_grad():
            batch_x = batch_x.to(self.device)
            outputs = self.bm_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompts)[:, :, 0:]
        return outputs