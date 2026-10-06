"""
BGE-M3 集成版向量 RAG 服务
- 使用 BGE-M3 作为 Embedder
- 支持 GPU 加速
- 集成 LangChain Embeddings 接口
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("⚠️ FAISS 未安装")


class BGE3Embedder:
    """
    BGE-M3 Embedder - 直接使用 Transformers
    支持 GPU 加速，1024 维
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        if model_path is None:
            # 默认路径
            base_dir = Path(__file__).parent.parent.parent
            model_path = str(base_dir / "app" / "graphrag" / "models" / "bge-m3" / 
                           "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181")
        
        self.model_path = model_path
        
        # 自动检测设备
        if device == "auto":
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            import torch
            self.device = torch.device(device)
        
        print(f"[BGE-M3] Loading model from: {model_path}")
        print(f"[BGE-M3] Using device: {self.device}")
        
        self._load_model()
        self.dimension = 1024
        print(f"[BGE-M3] Model loaded, dimension: {self.dimension}")
    
    def _load_model(self):
        """加载模型"""
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False
        )
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.F = F
        self.torch = torch
    
    def _mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return self.torch.sum(token_embeddings * input_mask_expanded, 1) / self.torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """批量编码文本"""
        import torch
        
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
                embeddings = self.F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.embed([text])[0]


class VectorRAG:
    """向量检索系统 - 使用 BGE-M3"""
    
    def __init__(
        self, 
        embedder: Optional[BGE3Embedder] = None,
        device: str = "auto",
        model_path: Optional[str] = None
    ):
        """
        初始化向量 RAG
        
        Args:
            embedder: 自定义 embedder 实例。如果为 None，则创建新的 BGE3Embedder
            device: 设备 ('cuda', 'cpu', 'auto')
            model_path: 模型路径
        """
        self.device = device
        
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = BGE3Embedder(model_path=model_path, device=device)
        
        self.faiss_index = None
        self.documents = []
        self.metadata = []
        
    def embed_documents(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """对文档批量编码"""
        if not texts:
            return np.array([])
        
        print(f"🚀 开始编码 {len(texts)} 个文档...")
        embeddings = self.embedder.embed(texts, batch_size=batch_size)
        print(f"✅ 编码完成: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """对查询编码"""
        return self.embedder.embed_single(query)
    
    def build_index(self, documents: List[str], metadata: List[Dict] = None):
        """构建向量索引"""
        if not documents:
            raise ValueError("没有文档可索引")
        
        print(f"🔨 开始构建索引：{len(documents)} 个文档")
        
        # 编码文档
        embeddings = self.embed_documents(documents)
        
        # 创建 FAISS 索引
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        
        # 保存文档和元数据
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]
        
        print(f"✅ 索引构建完成：{self.faiss_index.ntotal} 个向量")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, str, float]]:
        """搜索最相关的文档片段"""
        if self.faiss_index is None:
            raise ValueError("索引未构建，请先调用 build_index")
        
        # 编码查询
        query_embedding = self.embed_query(query)
        query_embedding = np.array([query_embedding])
        
        # 搜索
        scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.documents)))
        
        # 返回结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.documents):
                doc_meta = self.metadata[idx] if idx < len(self.metadata) else {}
                results.append((doc_meta, self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, output_dir: str):
        """保存索引到磁盘"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 FAISS 索引
        faiss.write_index(self.faiss_index, os.path.join(output_dir, "faiss.index"))
        
        # 保存 embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), 
                np.array([self.embed_query(d) for d in self.documents]))
        
        # 保存文档和元数据
        with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
            import pickle
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "dimension": self.embedder.dimension,
                "encoder": "bge-m3"
            }, f)
        
        print(f"💾 索引已保存到：{output_dir}")
    
    def load_index(self, input_dir: str):
        """从磁盘加载索引"""
        import pickle
        
        index_path = os.path.join(input_dir, "faiss.index")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在：{input_dir}")
        
        # 加载 FAISS 索引
        self.faiss_index = faiss.read_index(index_path)
        
        # 加载文档
        with open(os.path.join(input_dir, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        
        print(f"✅ 索引已加载：{self.faiss_index.ntotal} 个向量")


# 便捷函数
def create_vector_rag(model_path: Optional[str] = None, device: str = "auto") -> VectorRAG:
    """创建 BGE-M3 向量 RAG 实例"""
    return VectorRAG(model_path=model_path, device=device)


if __name__ == "__main__":
    # 测试
    print("Testing BGE-M3 Vector RAG...")
    
    rag = VectorRAG()
    
    # 测试文档
    documents = [
        "博士研究生基准学制为4年",
        "硕士研究生基准学制为3年",
        "论文答辩需要提交论文"
    ]
    
    rag.build_index(documents)
    
    # 测试搜索
    results = rag.search("博士学制几年？", top_k=2)
    
    print("\nSearch results:")
    for i, (meta, doc, score) in enumerate(results, 1):
        print(f"  [{i}] {score:.4f}: {doc[:50]}...")