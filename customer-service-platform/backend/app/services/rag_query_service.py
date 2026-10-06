"""
RAG 查询服务
集成 BGE-M3 向量检索 + DeepSeek 生成式回复
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any

from opentelemetry import trace


# 向量 RAG 单例
_rag_instance = None


def get_rag_instance(user_id: str = "8e8dca41-982b-53a3-b298-7d6a2b0ed7ee"):
    """获取或创建向量 RAG 实例"""
    global _rag_instance
    
    if _rag_instance is None:
        from app.services.vector_rag_bge_m3 import VectorRAG
        _rag_instance = VectorRAG()
        
        # 加载已索引的文档
        output_dir = Path(f"app/graphrag/data/output/{user_id}/vector_rag")
        metadata_path = output_dir / "metadata_bgem3.pkl"
        
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                meta = pickle.load(f)
            chunks = meta["chunks"]
            _rag_instance.build_index(chunks)
            print(f"[RAG] 已加载 {len(chunks)} 个文档块")
    
    return _rag_instance


class RAGQueryService:
    """RAG 查询服务"""
    
    def __init__(self, user_id: str = "8e8dca41-982b-53a3-b298-7d6a2b0ed7ee"):
        self.user_id = user_id
        self.rag = get_rag_instance(user_id)
        
        # DeepSeek 客户端（如果配置了）
        self.deepseek = None
        if os.getenv("DEEPSEEK_API_KEY"):
            from app.services.deepseek_service import DeepSeekClient
            self.deepseek = DeepSeekClient()
        
        self.tracer = trace.get_tracer("rag_query")
        
        # 数据飞轮
        from app.services.data_flywheel import DataFlywheel
        self.flywheel = DataFlywheel(auto_log_threshold=0.65)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关文档块
        """
        with self.tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)
            
            results = self.rag.search(query, top_k=top_k * 2)
            
            chunks = []
            seen = set()
            for meta, doc, score in results:
                key = doc[:50]
                if key not in seen:
                    seen.add(key)
                    chunks.append({
                        "content": doc,
                        "score": float(score),
                        "source": meta.get("source", "unknown") if isinstance(meta, dict) else "unknown",
                        "chunk_index": len(chunks)
                    })
                    if len(chunks) >= top_k:
                        break
            
            span.set_attribute("num_results", len(chunks))
            return chunks[:top_k]
    
    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """
        使用 DeepSeek 生成回复
        """
        if not self.deepseek:
            return self._fallback_response(context_chunks)
        
        with self.tracer.start_as_current_span("rag.generate") as span:
            context = "\n\n".join([c["content"] for c in context_chunks[:3]])
            
            prompt = f"""你是一个专业的学术助手，请基于以下上下文回答问题。

查询: {query}

参考资料:
{context}

请给出简洁准确的答案。"""
            
            span.set_attribute("prompt_length", len(prompt))
            
            response = self.deepseek.chat([{"role": "user", "content": prompt}])
            span.set_attribute("response_length", len(response))
            
            return response
    
    def _fallback_response(self, context_chunks: List[Dict]) -> str:
        """无 LLM 时的备用响应"""
        if not context_chunks:
            return "抱歉，未找到相关知识。"
        
        response_parts = []
        for i, chunk in enumerate(context_chunks[:3], 1):
            preview = chunk["content"][:200].replace('\n', ' ').strip()
            if preview:
                response_parts.append(f"{i}. {preview}...")
        
        return "找到以下相关内容：\n\n" + "\n".join(response_parts)
    
    def query(self, query_text: str, top_k: int = 5) -> Dict:
        """
        完整的 RAG 查询流程（含数据飞轮）
        """
        with self.tracer.start_as_current_span("rag.query") as span:
            span.set_attribute("query", query_text)
            
            # 1. 检索
            chunks = self.retrieve(query_text, top_k)
            
            # 2. 数据飞轮: 尝试已补充的知识
            resolved_knowledge = self.flywheel.get_resolved_knowledge(query_text)
            if resolved_knowledge:
                span.set_attribute("flywheel_hit", True)
                # 飞轮命中时直接给出补充知识答案
                return {
                    "query": query_text,
                    "results": chunks,
                    "answer": "【数据飞轮】根据补充知识：" + "；".join(resolved_knowledge),
                    "total_results": len(chunks),
                    "confidence": 0.95,
                    "flywheel_hit": True
                }
            
            # 3. 生成
            answer = self.generate(query_text, chunks)
            
            # 4. 数据飞轮: 记录低置信度回答
            top_score = chunks[0]["score"] if chunks else 0.0
            confidence = min(0.95, top_score + 0.15)  # 简单映射到置信度
            
            if self.flywheel.should_log(confidence):
                self.flywheel.log_failure(
                    query=query_text,
                    answer=answer,
                    confidence=confidence,
                    failure_type="low_confidence",
                    metadata={"top_score": top_score, "num_results": len(chunks)}
                )
            
            span.set_attribute("num_results", len(chunks))
            span.set_attribute("confidence", confidence)
            
            return {
                "query": query_text,
                "results": chunks,
                "answer": answer,
                "total_results": len(chunks),
                "confidence": confidence,
                "flywheel_hit": False
            }


def query_rag(query: str, top_k: int = 5, user_id: str = None) -> Dict:
    """
    快速查询函数
    """
    service = RAGQueryService(user_id or "8e8dca41-982b-53a3-b298-7d6a2b0ed7ee")
    return service.query(query, top_k)


if __name__ == "__main__":
    # 测试
    print("测试 RAG 服务...")
    
    results = query_rag("博士学制几年？", top_k=3)
    print(f"查询: {results['query']}")
    print(f"结果: {results['total_results']} 个")
    for i, r in enumerate(results['results'][:2], 1):
        print(f"  [{i}] {r['score']:.4f}: {r['content'][:50]}...")