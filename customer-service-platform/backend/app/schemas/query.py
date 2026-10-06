"""查询相关请求/响应 Schema"""
from typing import List, Dict, Optional
from pydantic import BaseModel


class RAGQueryRequest(BaseModel):
    """RAG 查询请求"""
    query: str
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None
    top_k: int = 5


class RAGQueryResponse(BaseModel):
    """RAG 查询响应"""
    query: str
    results: List[Dict[str, str]]
    total_results: int


class DocumentChunk(BaseModel):
    """文档块"""
    content: str
    score: float
    source: Optional[str] = None
    chunk_index: Optional[int] = None