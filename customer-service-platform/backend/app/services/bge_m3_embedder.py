#!/usr/bin/env python3
"""
BGE-M3 LangChain Embedder 集成模块
用于生产环境的向量检索服务
"""
import os
import sys
import numpy as np
from typing import List, Optional
from pathlib import Path
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class BGEM3Embeddings(Embeddings):
    """
    BGE-M3 LangChain Embedder
    
    使用 Transformers 原生加载 BGE-M3 模型，支持 GPU 加速
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化 BGE-M3 Embedder
        
        Args:
            model_path: 模型本地路径。如果为 None，则使用默认路径
            device: 运行设备 ('cuda' 或 'cpu')。如果为 None，则自动检测
        """
        if model_path is None:
            # 默认模型路径
            base_dir = Path(__file__).parent.parent.parent
            model_path = str(base_dir / "app" / "graphrag" / "models" / "bge-m3" / 
                           "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181")
        
        self.model_path = model_path
        
        # 自动检测设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[BGE-M3] Loading model from: {model_path}")
        print(f"[BGE-M3] Using device: {self.device}")
        
        # 加载模型
        self._load_model()
        
        self.dimension = 1024
        print(f"[BGE-M3] Model loaded successfully, dimension: {self.dimension}")
    
    def _load_model(self):
        """加载 BGE-M3 模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False
        )
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            numpy 数组，形状为 (len(texts), 1024)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded)
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为一组文档生成嵌入向量
        
        Args:
            texts: 文档文本列表
            
        Returns:
            嵌入向量列表，每个向量长度为 1024
        """
        embeddings = self._encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        为查询文本生成嵌入向量
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量，长度为 1024
        """
        embedding = self._encode([text])
        return embedding[0].tolist()


# 单例实例
_bge_m3_instance: Optional[BGEM3Embeddings] = None

def get_bge_m3_embedder(model_path: Optional[str] = None) -> BGEM3Embeddings:
    """
    获取 BGE-M3 Embedder 单例
    
    Args:
        model_path: 可选的自定义模型路径
        
    Returns:
        BGEM3Embeddings 实例
    """
    global _bge_m3_instance
    
    if _bge_m3_instance is None or model_path is not None:
        _bge_m3_instance = BGEM3Embeddings(model_path)
    
    return _bge_m3_instance


# 便捷函数
def embed_texts(texts: List[str], model_path: Optional[str] = None) -> List[List[float]]:
    """
    快速嵌入函数
    
    Args:
        texts: 文本列表
        model_path: 可选的自定义模型路径
        
    Returns:
        嵌入向量列表
    """
    embedder = get_bge_m3_embedder(model_path)
    return embedder.embed_documents(texts)


def embed_text(text: str, model_path: Optional[str] = None) -> List[float]:
    """
    快速嵌入单个文本
    
    Args:
        text: 文本
        model_path: 可选的自定义模型路径
        
    Returns:
        嵌入向量
    """
    embedder = get_bge_m3_embedder(model_path)
    return embedder.embed_query(text)


if __name__ == "__main__":
    # 测试
    print("Testing BGE-M3 Embedder...")
    
    embedder = BGEM3Embeddings()
    
    # 测试嵌入
    texts = ["博士学制四年", "硕士论文答辩要求"]
    embeddings = embedder.embed_documents(texts)
    
    print(f"Embedded {len(texts)} texts")
    print(f"Embedding shape: {len(embeddings)}, {len(embeddings[0])}")
    
    # 测试查询
    query_emb = embedder.embed_query("学制是多少年？")
    print(f"Query embedding shape: {len(query_emb)}")