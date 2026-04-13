# core/reranker.py
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 关闭离线模式！！
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import torch
from typing import List, Tuple
from sentence_transformers import CrossEncoder
# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import logger

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"加载 Rerank 模型: {model_name}, 设备: {self.device}")
        # 使用 CrossEncoder 替代 FlagReranker
        self.model = CrossEncoder(model_name, device=self.device)
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
        """
        对文档列表进行重排序
        :param query: 用户查询
        :param documents: 文档内容列表
        :param top_k: 返回最相关的前 k 个索引
        :return: [(index, score), ...] 按分数降序排列
        """
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        # CrossEncoder.predict 返回 numpy 数组，分数越高表示越相关
        scores = self.model.predict(pairs)
        # 转换为 Python 列表
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        # 排序并取前 top_k
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return scored[:top_k]