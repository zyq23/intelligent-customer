#!/usr/bin/env python3
"""
高级 RAG 系统 v3.0
集成优化：
1. BGE-M3 作为主要 Embedding 模型
2. BM25 + 向量混合搜索
3. Cross Encoder 重排序
4. 查询扩展
5. DeepSeek 模型自动切换
"""

import os, sys, json, pickle, numpy as np, faiss, re, requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

# ========================================
# 配置管理
# ========================================
USER_ID = os.getenv("RAG_USER_ID", "8e8dca41-982b-53a3-b298-7d6a2b0ed7ee")

# Ollama 配置（备用）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:21434")

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 模型路径
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "graphrag" / "models"
RAG_OUTPUT_DIR = BASE_DIR / "graphrag" / "data" / "output" / USER_ID / "vector_rag"

# ========================================
# 工具函数
# ========================================
def load_json_env(env_path: str = ".env"):
    """加载环境变量"""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip().strip('"')
    
    # 更新系统环境变量
    for k, v in env_vars.items():
        os.environ[k] = v
    
    return env_vars

# 加载环境变量
load_json_env()


# ========================================
# DeepSeek 客户端
# ========================================
class DeepSeekClient:
    """DeepSeek API 客户端，支持自动降级"""
    
    def __init__(self, max_tokens: int = 2048, temperature: float = 0.3):
        self.api_key = DEEPSEEK_API_KEY
        self.base_url = DEEPSEEK_BASE_URL.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 模型列表（优先级从高到低）
        self.models = [
            ("deepseek-v3-pro-0813", True),   # Pro 版本
            ("deepseek-chat", True),         # V3
            ("deepseek-coder", True),        # Coder
            ("deepseek-v2-flash", False),    # Flash 版本
            ("deepseek-v2-lite", False),
        ]
        
        self.current_model_index = 0
        self.failed_requests = 0
        self.max_failures_before_switch = 3
    
    def _get_current_model(self):
        """获取当前模型名称"""
        model_name, _ = self.models[self.current_model_index % len(self.models)]
        return model_name
    
    def is_flash_model(self):
        """检查当前是否是 Flash 模型"""
        _, is_pro = self.models[self.current_model_index % len(self.models)]
        return not is_pro
    
    def switch_to_next_model(self):
        """切换到下一个可用模型"""
        self.current_model_index += 1
        model_name, is_pro = self.models[self.current_model_index % len(self.models)]
        print(f"  ⚡ Switched to model: {model_name}")
        return model_name
    
    def chat(self, messages: List[Dict], retry_count: int = 3) -> str:
        """发送聊天请求，支持失败重试和模型降级"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        last_error = None
        
        for attempt in range(retry_count):
            try:
                model_name = self._get_current_model()
                
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # 重置失败计数
                    self.failed_requests = 0
                    
                    # 检查是否是 flash 模型的配额错误
                    if "quota" in str(response.text).lower() or "rate_limit" in str(response.text).lower():
                        if self.is_flash_model():
                            print(f"  ⚠️  Flash model quota exceeded, switching to Pro...")
                            self.switch_to_next_model()
                    
                    return content.strip()
                
                elif response.status_code == 429:
                    # 配额不足，尝试切换模型
                    self.failed_requests += 1
                    if self.failed_requests >= self.max_failures_before_switch:
                        print(f"  ⚡ Rate limit detected, switching model...")
                        self.switch_to_next_model()
                        last_error = "Rate limit - switched model"
                        continue
                    
                elif response.status_code >= 500:
                    # 服务器错误，短暂等待
                    time.sleep(1)
                
                last_error = f"HTTP {response.status_code}: {response.text[:100]}"
                
            except requests.exceptions.Timeout:
                last_error = "Timeout"
            except Exception as e:
                last_error = str(e)
        
        raise Exception(f"All retries failed: {last_error}")


# ========================================
# BGE-M3 Embedding 模型
# ========================================
class BGEM3Embedder:
    """BGE-M3 Embedding 包装器"""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(MODELS_DIR / "bge-m3" / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181")
        
        self.model_path = model_path
        self.model = None
        self.dimension = 1024  # BGE-M3 默认维度
        
        self._load_model()
    
    def _load_model(self):
        """加载 BGE-M3 模型"""
        from transformers import AutoModel, AutoTokenizer
        import torch
        import torch.nn.functional as F
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"BGE-M3 model not found at {self.model_path}")
        
        # 使用 Transformers 加载（slow tokenizer for sentencepiece）
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False
        )
        self.model = AutoModel.from_pretrained(self.model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.dimension = 1024  # BGE-M3 固定 1024 维
        self.use_sentence_transformers = False
        
        print(f"✅ BGE-M3 loaded on {self.device}, dimension: {self.dimension}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # BGE-M3 是 XLM-RoBERTa，1024 维
        self.dimension = 1024
        
        print(f"✅ BGE-M3 loaded on {self.device}, dimension: {self.dimension}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_queries(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        """编码查询"""
        import torch
        import torch.nn.functional as F
        
        all_embeddings = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            
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
    
    def encode_corpus(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        """编码文档"""
        return self.encode_queries(documents, batch_size=batch_size)
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode_queries([text])[0]
    
    def encode_queries(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        """编码查询"""
        return self.model.encode(queries, batch_size=batch_size, convert_to_numpy=True)
    
    def encode_corpus(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        """编码文档"""
        return self.model.encode(documents, batch_size=batch_size, convert_to_numpy=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode_queries([text])[0]


# ========================================
# BM25 检索器
# ========================================
class BM25Retriever:
    """BM25 关键字检索器"""
    
    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
        except ImportError:
            raise ImportError("Please install rank-bm25: pip install rank-bm25")
        
        self.bm25 = None
        self.tokenized_corpus = None
    
    def fit(self, documents: List[str]):
        """训练 BM25 模型"""
        # 简单的中文分词（空格分词 + 标点分隔）
        def tokenize(text: str) -> List[str]:
            # 替换中文标点和特殊符号为空格
            text = re.sub(r'[，。！？；：、（）《》【】""''“”…—\d]', ' ', text)
            tokens = text.lower().split()
            return [t for t in tokens if len(t) > 1]
        
        tokenized_corpus = [tokenize(doc) for doc in documents]
        self.bm25 = self.BM25Okapi(tokenized_corpus)
        self.tokenized_corpus = tokenized_corpus
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """搜索最相关的文档"""
        def tokenize(text: str) -> List[str]:
            text = re.sub(r'[，。！？；：、（）《》【】""''“”…—\d]', ' ', text)
            tokens = text.lower().split()
            return [t for t in tokens if len(t) > 1]
        
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k
        top_indices = scores.argsort()[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        return results


# ========================================
# Cross Encoder Reranker
# ========================================
class BGEReranker:
    """BGE Reranker 重排序"""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(MODELS_DIR / "bge-reranker-large")
        
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        """加载 reranker 模型"""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float32
        )
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✅ Reranker loaded on {self.device}")
    
    def rerank(self, query: str, passages: List[str], top_n: int = 5) -> List[Tuple[int, float]]:
        """重排序文档"""
        if not hasattr(self, 'model'):
            # 如果没有 reranker，返回空列表
            return []
        
        import torch
        from torch.nn.functional import softmax
        
        pairs = [[query, passage] for passage in passages]
        
        # 分批处理
        batch_size = 8
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            
            # 使用 HuggingFace Transformers
            inputs = self.model_tokenizer.batch_encode_plus(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            
            all_scores.extend(scores.tolist())
        
        # 按分数排序
        scored_indices = [(i, score) for i, score in enumerate(all_scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        return scored_indices[:top_n]
    
    def rerank_simple(self, query: str, passages: List[str], top_n: int = 5) -> List[Tuple[int, float]]:
        """简化版 rerank（不使用完整模型，使用相似度）"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_path)
            
            encoded_input = model.encode([query] + passages, batch_size=8)
            query_emb = encoded_input[0]
            passage_embs = encoded_input[1:]
            
            # 余弦相似度
            scores = []
            for i, emb in enumerate(passage_embs):
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                scores.append((i, sim))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_n]
        
        except Exception as e:
            print(f"  Rerank fallback: {e}")
            # 回退到无 rerank
            return []


# ========================================
# 混合检索器
# ========================================
class HybridRetriever:
    """混合检索：向量 + BM25"""
    
    def __init__(self, embedder: BGEM3Embedder, bm25: BM25Retriever = None, alpha: float = 0.7):
        """
        Args:
            embedder: 向量模型
            bm25: BM25 模型
            alpha: 融合权重 (0.7 = 70% 向量 + 30% BM25)
        """
        self.embedder = embedder
        self.bm25 = bm25
        self.alpha = alpha
    
    def fuse_scores(self, vector_scores: List[float], bm25_scores: List[Tuple[int, float]]) -> Dict[int, float]:
        """归一化并融合分数"""
        # 归一化向量分数到 0-1
        if vector_scores:
            min_score = min(vector_scores)
            max_score = max(vector_scores)
            if max_score > min_score:
                normalized_vector = {(i, (s - min_score) / (max_score - min_score)) 
                                   for i, s in enumerate(vector_scores)}
            else:
                normalized_vector = {i: 0.5 for i, s in enumerate(vector_scores)}
        else:
            normalized_vector = {}
        
        # 归一化 BM25 分数
        if bm25_scores:
            scores_only = [s for _, s in bm25_scores]
            min_bm25 = min(scores_only)
            max_bm25 = max(scores_only)
            if max_bm25 > min_bm25:
                normalized_bm25 = {i: (s - min_bm25) / (max_bm25 - min_bm25) 
                                  for i, s in bm25_scores}
            else:
                normalized_bm25 = {i: 0.5 for i, _ in bm25_scores}
        else:
            normalized_bm25 = {}
        
        # 加权融合
        fused = {}
        all_indices = set(normalized_vector.keys()) | set(normalized_bm25.keys())
        
        for idx in all_indices:
            vec_score = normalized_vector.get(idx, 0)
            bm25_score = normalized_bm25.get(idx, 0)
            fused[idx] = self.alpha * vec_score + (1 - self.alpha) * bm25_score
        
        return fused
    
    def search(self, query: str, chunks: List[str], top_k: int = 20, 
               return_details: bool = False) -> List[Dict]:
        """混合搜索"""
        
        # 1. 向量检索
        query_emb = self.embedder.encode_single(query)
        
        # 批量编码文档
        docs_embs = self.embedder.encode_corpus(chunks)
        
        # 余弦相似度
        similarities = []
        for i, doc_emb in enumerate(docs_embs):
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            similarities.append(sim)
        
        # 2. BM25 检索
        bm25_results = []
        if self.bm25:
            bm25_results = self.bm25.search(query, top_k=min(len(chunks), 50))
        
        # 3. 融合分数
        fused_scores = self.fuse_scores(similarities, bm25_results)
        
        # 4. 排序
        ranked_indices = sorted(fused_scores.keys(), 
                              key=lambda i: fused_scores[i], 
                              reverse=True)[:top_k]
        
        results = []
        for idx in ranked_indices:
            results.append({
                "content": chunks[idx],
                "similarity_score": float(fused_scores[idx]),
                "chunk_idx": idx
            })
        
        return results


# ========================================
# 查询扩展
# ========================================
class QueryExpander:
    """使用 LLM 扩展查询"""
    
    def __init__(self, client: DeepSeekClient = None):
        self.client = client or DeepSeekClient(max_tokens=512, temperature=0.3)
    
    def expand(self, query: str, num_expansions: int = 3) -> List[str]:
        """扩展查询"""
        
        prompt = f"""你是一个专业的学术助手。请根据以下用户问题，生成 3-5 个不同表达方式的同义问题，帮助更好地检索相关信息。

用户问题：{query}

请按照以下格式输出：
同义问题 1: ...
同义问题 2: ...
同义问题 3: ...

注意：
- 保持问题的原意不变
- 使用不同的词汇和句式
- 包含可能的关键词变体"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat(messages)
            
            # 解析响应
            expansions = []
            lines = response.strip().split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        exp = parts[1].strip()
                        if exp and len(exp) > 5:
                            expansions.append(exp)
            
            # 如果解析失败，返回原始查询
            if not expansions:
                expansions = [query]
            
            # 最多返回 num_expansions 个
            return expansions[:num_expansions]
        
        except Exception as e:
            print(f"  Query expansion failed: {e}")
            return [query]


# ========================================
# 主 RAG 类
# ========================================
class AdvancedRAG:
    """高级 RAG 系统"""
    
    def __init__(self, user_id: str = None, use_hybrid: bool = True, 
                 use_reranker: bool = True, use_query_expansion: bool = True):
        self.user_id = user_id or USER_ID
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.use_query_expansion = use_query_expansion
        
        self.embedder = None
        self.bm25 = None
        self.reranker = None
        self.expander = None
        self.hybrid_retriever = None
        
        self.chunks = []
        self.metadata = {}
        
        self._load_components()
    
    def _load_components(self):
        """加载所有组件"""
        print("="*60)
        print("Advanced RAG System v3.0 - Loading Components")
        print("="*60)
        
        # 1. 加载 Embedder
        print("\n[1] Loading BGE-M3 Embedder...")
        try:
            self.embedder = BGEM3Embedder()
            print("✅ BGE-M3 loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load BGE-M3: {e}")
            raise
        
        # 2. 加载 metadata
        print("\n[2] Loading document metadata...")
        try:
            # Try new BGE-M3 metadata first
            metadata_file = RAG_OUTPUT_DIR / "metadata_bge_m3.pkl"
            if not metadata_file.exists():
                metadata_file = RAG_OUTPUT_DIR / "metadata_bgem3.pkl"
                if not metadata_file.exists():
                    metadata_file = RAG_OUTPUT_DIR / "metadata.pkl"
            
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
                self.chunks = self.metadata.get("chunks", [])
            
            print(f"✅ Loaded {len(self.chunks)} chunks (encoder: {self.metadata.get('encoder', 'unknown')})")
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
            raise
        
        # 3. 加载 BM25
        if self.use_hybrid:
            print("\n[3] Building BM25 index...")
            self.bm25 = BM25Retriever()
            self.bm25.fit(self.chunks)
            print("✅ BM25 index built")
        
        # 4. 加载 Reranker
        if self.use_reranker:
            print("\n[4] Loading BGE Reranker...")
            try:
                self.reranker = BGEReranker()
                print("✅ Reranker loaded")
            except Exception as e:
                print(f"⚠️  Failed to load reranker: {e}")
                self.use_reranker = False
        
        # 5. 初始化 Query Expander
        if self.use_query_expansion:
            print("\n[5] Initializing Query Expander...")
            self.expander = QueryExpander()
            print("✅ Query Expander ready")
        
        # 6. 初始化 Hybrid Retriever
        if self.use_hybrid and self.bm25:
            self.hybrid_retriever = HybridRetriever(self.embedder, self.bm25, alpha=0.7)
            print("✅ Hybrid retriever ready")
        
        print("\n" + "="*60)
        print("✅ All components loaded!")
        print("="*60)
    
    def query(self, query: str, top_k: int = 10) -> List[Dict]:
        """执行查询"""
        
        # 1. 查询扩展（可选）
        queries_to_search = [query]
        if self.use_query_expansion and self.expander:
            expanded_queries = self.expander.expand(query)
            queries_to_search.extend(expanded_queries)
        
        all_results = []
        
        # 2. 对每个查询进行搜索
        for q in queries_to_search:
            if self.use_hybrid and self.hybrid_retriever:
                # 混合搜索
                results = self.hybrid_retriever.search(q, self.chunks, top_k=top_k*2)
            else:
                # 纯向量搜索
                query_emb = self.embedder.encode_single(q)
                docs_embs = self.embedder.encode_corpus(self.chunks)
                
                similarities = []
                for i, doc_emb in enumerate(docs_embs):
                    sim = np.dot(query_emb, doc_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                    )
                    similarities.append((i, float(sim)))
                
                results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k*2]
                results = [{"content": self.chunks[i], "similarity_score": s, "chunk_idx": i} 
                          for i, s in results]
            
            all_results.extend(results)
        
        # 3. 去重
        seen = set()
        unique_results = []
        for r in all_results:
            content = r["content"][:100]  # 使用前 100 字符作为唯一标识
            if content not in seen:
                seen.add(content)
                unique_results.append(r)
        
        # 4. 重排序（可选）
        if self.use_reranker and self.reranker and len(unique_results) > top_k:
            passages = [r["content"] for r in unique_results[:50]]
            reranked = self.reranker.rerank_simple(query, passages, top_n=top_k)
            
            if reranked:
                # 重新排序
                reranked_content = {i: score for i, score in reranked}
                unique_results = [r for i, r in enumerate(unique_results[:50]) 
                                if i in reranked_content]
                # 按 rerank 分数排序
                unique_results.sort(
                    key=lambda r: reranked_content.get(self.chunks.index(r["content"]), 0),
                    reverse=True
                )
        
        # 5. 返回 top-k
        return unique_results[:top_k]
    
    def rebuild_index(self):
        """重新构建索引（使用 BGE-M3）"""
        print("\n🔄 Rebuilding index with BGE-M3...")
        
        # 1. 编码所有 chunks
        embeddings = self.embedder.encode_corpus(self.chunks)
        
        # 2. 创建 FAISS 索引
        index = faiss.IndexFlatIP(self.embedder.dimension)
        # 归一化
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype(np.float32))
        
        # 3. 保存
        output_files = {
            "faiss.index": index,
            "embeddings.npy": embeddings,
            "metadata_bgem3.pkl": {**self.metadata, "dimension": self.embedder.dimension}
        }
        
        for name, data in output_files.items():
            filepath = RAG_OUTPUT_DIR / name
            if name.endswith(".index"):
                faiss.write_index(data, str(filepath))
            elif name.endswith(".npy"):
                np.save(str(filepath), data)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
        
        print(f"✅ Index rebuilt and saved")
        print(f"   - Dimension: {self.embedder.dimension}")
        print(f"   - Chunks: {len(self.chunks)}")


# ========================================
# CLI 接口
# ========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG Query Interface")
    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--hybrid", action="store_true", default=True, help="Use hybrid search")
    parser.add_argument("--rerank", action="store_true", default=True, help="Use reranker")
    parser.add_argument("--expand", action="store_true", default=True, help="Use query expansion")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index with BGE-M3")
    parser.add_argument("--dry-run", action="store_true", help="Test without actual query")
    
    args = parser.parse_args()
    
    # 初始化 RAG
    rag = AdvancedRAG(
        use_hybrid=args.hybrid,
        use_reranker=args.rerank,
        use_query_expansion=args.expand
    )
    
    if args.dry_run:
        print("✅ System initialized successfully!")
        print(f"   - Chunks: {len(rag.chunks)}")
        print(f"   - Dimension: {rag.embedder.dimension}")
        print(f"   - Hybrid: {rag.use_hybrid}")
        print(f"   - Reranker: {rag.use_reranker}")
        print(f"   - Query Expansion: {rag.use_query_expansion}")
    
    elif args.rebuild:
        rag.rebuild_index()
    
    elif args.query:
        print(f"\n🔍 Query: {args.query}")
        results = rag.query(args.query, top_k=args.top_k)
        
        print(f"\n{'='*70}")
        print(f"Results ({len(results)} found)")
        print('='*70)
        
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['similarity_score']:.4f}")
            print(f"    Content: {r['content'][:200]}...")
        
        print(f"\n{'='*70}")
    
    else:
        print("Usage: python advanced_rag.py 'your query' [--dry-run] [--rebuild]")
