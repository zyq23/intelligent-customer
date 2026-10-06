"""聊天相关请求/响应 Schema"""
from typing import List, Dict, Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """普通对话请求"""
    messages: List[Dict[str, str]]
    user_id: int
    conversation_id: int


class ReasonRequest(BaseModel):
    """深度思考请求"""
    messages: List[Dict[str, str]]
    user_id: int


class CreateConversationRequest(BaseModel):
    user_id: int


class UpdateConversationNameRequest(BaseModel):
    name: str


class LangGraphResumeRequest(BaseModel):
    query: str
    user_id: int
    conversation_id: str
