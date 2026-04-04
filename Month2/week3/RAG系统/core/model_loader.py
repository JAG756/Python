# core/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import MODEL_NAME

class ModelLoader:
    """模型加载器"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = MODEL_NAME
        self.tokenizer = None
        self.model = None
        
    def load(self):
        """加载模型（使用本地缓存）"""
        print(f"✅ 使用设备：{self.device}")
        print(f"📦 加载模型：{self.model_name}")
        print("📁 使用本地缓存，不联网...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            local_files_only=True  # 关键：只使用本地缓存
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True  # 关键：只使用本地缓存
        ).eval()
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print("✅ 模型加载完成")
        return self
    
    def generate(self, prompt, max_new_tokens, temperature):
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        answer = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return answer.strip()