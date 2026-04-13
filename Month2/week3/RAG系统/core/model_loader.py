# core/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import MODEL_NAME, MAX_PROMPT_LENGTH
from utils.logger import logger

class ModelLoader:
    """模型加载器"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = MODEL_NAME
        self.tokenizer = None
        self.model = None
        
    def load(self):
        """加载模型（使用本地缓存）"""
        logger.info(f"使用设备：{self.device}")
        logger.info(f"加载模型：{self.model_name}")
        logger.info("使用本地缓存，不联网...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception as e:
            logger.error(f"Tokenizer 加载失败: {e}")
            raise RuntimeError(
                f"模型 {self.model_name} 的 tokenizer 加载失败。\n"
                "请确认：\n"
                "1. 模型已下载到本地缓存目录\n"
                "2. 缓存目录路径正确\n"
                "3. 磁盘空间充足"
            ) from e
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=True
            ).eval()
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(
                f"模型 {self.model_name} 加载失败。\n"
                "可能原因：\n"
                "1. 模型文件不完整，请重新下载\n"
                "2. 内存/显存不足\n"
                "3. 模型版本与代码不兼容"
            ) from e
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        logger.info("模型加载完成")
        return self
    
    def generate(self, prompt, max_new_tokens, temperature):
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=MAX_PROMPT_LENGTH).to(self.device)
        
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