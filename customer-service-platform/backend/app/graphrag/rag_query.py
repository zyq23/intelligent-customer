#!/usr/bin/env python3
"""
简化版向量 RAG 查询接口
使用 Ollama nomic-embed-text + FAISS 实现语义搜索
"""
import os, sys, json, pickle, requests
import numpy as np, faiss
from pathlib import Path
from typing import List, Dict

# 配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:21434")
EMBEDDING_MODEL = "nomic-embed-text"

class SimpleVectorRAG:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.base_dir = Path(f"app/graphrag/data/output/{user_id}/vector_rag")
        self.index = None
        self.metadata = None
        
    def load(self):
        """加载 FAISS 索引和元数据"""
        index_file = self.base_dir / "faiss.index"
        meta_file = self.base_dir / "metadata.pkl"
        
        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"未找到 RAG 索引，请先运行索引构建")
            
        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            self.metadata = pickle.load(f)
            
    def embed(self, text: str) -> np.ndarray:
        """调用 Ollama embedding API"""
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=120
        )
        return np.array(resp.json()["embedding"])
    
    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        查询相关文档
        
        Args:
            question: 用户问题
            top_k: 返回最相关的 K 个片段
            
        Returns:
            包含相似度分数、内容和来源的字典列表
        """
        if self.index is None:
            self.load()
            
        # 编码问题
        q_emb = self.embed(question).reshape(1, -1)
        
        # 搜索
        scores, indices = self.index.search(q_emb, min(top_k, len(self.metadata["chunks"])))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "rank": len(results) + 1,
                "similarity_score": float(score),
                "content": self.metadata["chunks"][idx],
                "source": self.metadata["sources"][idx]
            })
            
        return results


def main():
    """命令行测试"""
    if len(sys.argv) < 2:
        print("用法：python rag_query.py '<你的问题>'")
        sys.exit(1)
        
    question = sys.argv[1]
    user_id = "8e8dca41-982b-53a3-b298-7d6a2b0ed7ee"
    
    try:
        rag = SimpleVectorRAG(user_id)
        results = rag.query(question, top_k=3)
        
        print(f"\n问题：{question}\n")
        print("-"*60)
        for r in results:
            print(f"[{r['rank']}] 相似度：{r['similarity_score']:.3f}")
            print(f"    来源：{r['source']}")
            print(f"    内容：{r['content'][:150]}...")
            print()
            
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
