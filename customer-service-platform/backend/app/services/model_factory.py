"""大模型实例工厂

集中管理 LLM 实例的创建逻辑, 避免在业务代码中散落 if/else 分支。
根据 .env 中的 AGENT_SERVICE 配置自动选择 DeepSeek(百炼兼容模式) 或 Ollama。
"""
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from app.core.config import settings, ServiceType
from app.core.logger import get_logger

logger = get_logger(service="model_factory")


def create_agent_llm(
    temperature: float = 0.7,
    tags: Optional[list[str]] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """根据 AGENT_SERVICE 配置创建 Agent 使用的大模型实例

    Args:
        temperature: 采样温度
        tags: LangChain 调用标签(用于追踪/观测)
        **kwargs: 透传给底层模型客户端的额外参数

    Returns:
        BaseChatModel: 配置好的大模型实例
    """
    tags = tags or []
    if settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
        logger.info(
            f"Creating ChatDeepSeek model: {settings.DEEPSEEK_MODEL} @ {settings.DEEPSEEK_BASE_URL}"
        )
        # 注意:
        # 1. ChatDeepSeek 的 base_url 字段名为 api_base, 必须显式传入, 否则会走到
        #    DeepSeek 官方域名导致 401
        # 2. 百炼 deepseek-v4-* 默认开启 thinking 模式, 该模式下不支持 tool_choice
        #    (with_structured_output 依赖), 这里显式关闭 thinking 以兼容工具调用
        return ChatDeepSeek(
            api_key=settings.DEEPSEEK_API_KEY,
            model_name=settings.DEEPSEEK_MODEL,
            api_base=settings.DEEPSEEK_BASE_URL,
            temperature=temperature,
            tags=tags,
            extra_body={"enable_thinking": False},
            **kwargs,
        )

    logger.info(
        f"Creating ChatOllama model: {settings.OLLAMA_AGENT_MODEL} @ {settings.OLLAMA_BASE_URL}"
    )
    return ChatOllama(
        model=settings.OLLAMA_AGENT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        tags=tags,
        **kwargs,
    )
