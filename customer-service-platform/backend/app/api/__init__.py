"""API 路由汇总"""
from fastapi import APIRouter

from app.api import agent, auth, chat, conversations, uploads, documents, query

api_router = APIRouter()

# 认证
api_router.include_router(auth.router)
# 聊天/推理/检索
api_router.include_router(chat.router)
# 会话管理
api_router.include_router(conversations.router)
# 文件/图片上传
api_router.include_router(uploads.router)
# 文档管理（知识库）
api_router.include_router(documents.router)
# LangGraph Agent
api_router.include_router(agent.router)
# RAG 查询
api_router.include_router(query.router)
