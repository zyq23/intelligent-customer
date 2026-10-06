"""问道星辰智能客服 - FastAPI 应用入口

企业级分层:
- app/api      : 路由层(HTTP 接口定义)
- app/core     : 核心配置/安全/日志
- app/services : 业务服务层(LLM/RAG/缓存/搜索)
- app/agent    : LangGraph Agent 编排层
- app/models   : SQLAlchemy 数据模型
- app/schemas  : Pydantic 数据校验
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.core.config import settings
from app.core.logger import get_logger
from app.core.middleware import LoggingMiddleware

# 初始化 OpenTelemetry tracing
try:
    from app.core.tracing import init_tracing, TRACING_ENABLED
    init_tracing("intelligent-customer")
except Exception as e:
    print(f"Warning: Tracing initialization failed: {e}")
    TRACING_ENABLED = False

logger = get_logger(service="main")

# 配置上传目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 创建 FastAPI 应用实例
app = FastAPI(
    title="问道星辰智能客服 REST API",
    version="2.0.0",
    description="基于 FastAPI + LangGraph + RAG 的企业级智能客服系统",
)

# 日志中间件
app.add_middleware(LoggingMiddleware)

# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请替换为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 业务路由统一挂载到 /api 前缀
app.include_router(api_router, prefix="/api")


@app.get("/health", tags=["system"])
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "intelligent-customer-service"}


# 前端静态资源(最后挂载, 避免覆盖 API 路由)
# 前端构建产物位于 backend/static/dist (Vue3 打包输出)
STATIC_DIR = Path(__file__).parent.parent / "static" / "dist"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
