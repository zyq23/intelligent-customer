"""
简化版向量 RAG - 基于 FAISS + Ollama bge-m3 embedding
绕过 GraphRAG 的实体提取问题，直接支持中文文档检索
"""

import os
import json
import asyncio
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("警告：FAISS 未安装，将使用 numpy 实现")


class OllamaEmbedder:
    """使用 Ollama 的 embedding API"""
    
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.dimension = 1024  # bge-m3 的维度
        
    def embed(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        embeddings = []
        
        for text in texts:
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=60
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            except Exception as e:
                print(f"❌ Embedding 错误：{e}")
                # 返回零向量作为 fallback
                embeddings.append([0.0] * self.dimension)
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.embed([text])[0]


class VectorRAG:
    """简化版向量检索系统"""
    
    def __init__(self, embedder: Optional[OllamaEmbedder] = None, device: str = "cpu"):
        self.device = device
        self.embedder = embedder or OllamaEmbedder()
        self.faiss_index = None
        self.documents = []  # 存储原始文本
        self.metadata = []   # 存储元数据
        
    def embed_documents(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """对文档批量编码"""
        if not texts:
            return np.array([])
        
        print(f"🚀 开始编码 {len(texts)} 个文档...")
        embeddings = self.embedder.embed(texts)
        print(f"✅ 编码完成")
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
        self.faiss_index = faiss.IndexFlatIP(dimension)  # 内积相似度
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
            if idx < len(self.documents):
                doc_meta = self.metadata[idx] if idx < len(self.metadata) else {}
                results.append((doc_meta, self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, output_dir: str):
        """保存索引到磁盘"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 FAISS 索引
        faiss.write_index(self.faiss_index, os.path.join(output_dir, "index.faiss"))
        
        # 保存文档和元数据
        with open(os.path.join(output_dir, "documents.json"), "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 索引已保存到：{output_dir}")
    
    def load_index(self, input_dir: str):
        """从磁盘加载索引"""
        index_path = os.path.join(input_dir, "index.faiss")
        docs_path = os.path.join(input_dir, "documents.json")
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(f"索引文件不存在：{input_dir}")
        
        # 加载 FAISS 索引
        self.faiss_index = faiss.read_index(index_path)
        
        # 加载文档
        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        
        print(f"✅ 索引已加载：{self.faiss_index.ntotal} 个向量")


class SimpleChunkingStrategy:
    """简单的文本分块策略"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """按字符数分块，尝试在句子边界断开"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句子边界断开
            if end < len(text):
                # 查找句号、问号、感叹号
                for sep in ['。\n', '。\n\n', '?\n', '!\n', '\n\n', '\n']:
                    pos = text.find(sep, start + chunk_size // 2, end + 200)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def split_by_chapters(text: str) -> List[Tuple[str, str]]:
        """按章节拆分文档，返回 (章节标题，内容) 列表"""
        import re
        
        chapters = []
        lines = text.split('\n')
        
        current_title = "全文"
        current_content = []
        
        chapter_pattern = re.compile(r'^[第\d][章章节节][^\n]*|^[\d]+\.|^一、|^二、|^三、')
        
        for line in lines:
            match = chapter_pattern.match(line.strip())
            if match and current_content:
                # 保存当前章节
                chapters.append((current_title, '\n'.join(current_content)))
                current_content = []
                current_title = line.strip()
            
            current_content.append(line)
        
        # 保存最后一章
        if current_content:
            chapters.append((current_title, '\n'.join(current_content)))
        
        return chapters if chapters else [("全文", text)]
