import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

llm_name_dict = {
    "llama3.1-8b": "/data2/user2/Llama-3.1-8B",
    "qwen3-1.7b": "/data2/user2/Qwen3-1.7B",
    "qwen3-8b": "/data2/user2/Qwen3-8B",
    "llama3.2-1b": "/data2/user2/Llama-3.2-1B/",
    "llama3.2-3b": "/data2/user2/Llama-3.2-3B/",
    "qwen3-4b": "/data2/user2/Qwen3-4B/"
}

class TextReinforcementModel():
    def __init__(self, llm_name="llama-3b-8b", llm_path="/data2/user2/Qwen3-1.7B/"):
        self.llm_name = llm_name
        if llm_path == "original":
            self.llm_path = llm_name_dict[llm_name.lower()]
        else:
            self.llm_path = llm_path
        
        # 检查模型路径是否存在
        import os
        if not os.path.exists(self.llm_path):
            raise ValueError(f"模型路径不存在: {self.llm_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side="left"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            for i, (name, param) in enumerate(self.model.named_parameters()): # 冻结LLM
                param.requires_grad = False
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def get_model_response(self, prompts,
                           max_new_tokens=1024,
                           do_sample=True,
                           temperature=1.5,
                           top_p=0.9):
        with torch.no_grad():
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=5120,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                use_cache=True
            )
            input_length = inputs.input_ids.size(1)
            generated_texts = self.tokenizer.batch_decode(
                outputs[:, input_length:],
                skip_special_tokens=True
            )

            return generated_texts