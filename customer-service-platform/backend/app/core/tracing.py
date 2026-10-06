"""
OpenTelemetry 全链路追踪中间件
自动追踪聊天对话、RAG 检索、DeepSeek 调用等关键操作
"""

import os
import time
from functools import wraps
from typing import Dict, Any, Optional
from contextlib import contextmanager

# 检查 OpenTelemetry 是否安装
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

# 是否启用追踪
TRACING_ENABLED = os.getenv("TRACING_ENABLED", "true").lower() == "true"
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")


def init_tracing(app_name: str = "intelligent-customer"):
    """
    初始化 OpenTelemetry 追踪
    
    Args:
        app_name: 应用名称
        
    Returns:
        TracerProvider 实例
    """
    if not HAS_OTEL or not TRACING_ENABLED:
        print("[Tracing] OpenTelemetry not available, skipping initialization")
        return None
    
    resource = Resource.create({
        "service.name": app_name,
        "service.version": "2.0.0",
        "deployment.environment": os.getenv("ENV", "production")
    })
    
    provider = TracerProvider(resource=resource)
    
    # 控制台导出 (开发调试)
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(
        BatchSpanProcessor(console_exporter)
    )
    
    # OTLP 导出 (生产环境)
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT)
        provider.add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )
    except Exception as e:
        print(f"[Tracing] OTLP exporter not available: {e}")
    
    trace.set_tracer_provider(provider)
    
    print(f"[Tracing] Initialized for {app_name}")
    return provider


def get_tracer(name: str = __name__):
    """获取 Tracer"""
    if not HAS_OTEL or not TRACING_ENABLED:
        return None
    
    return trace.get_tracer(name)


@contextmanager
def traced_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    创建追踪跨度
    
    Args:
        name: 跨度名称
        attributes: 属性字典
        
    Example:
        with traced_span("rag.search", {"query": "博士学制"}) as span:
            ...
    """
    tracer = get_tracer()
    
    if tracer is None:
        # 无追踪模式
        yield None
        return
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        yield span


def traced(func):
    """
    装饰器：自动追踪函数调用
    
    Example:
        @traced
        def embed_query(query: str) -> List[float]:
            ...
    """
    if not HAS_OTEL or not TRACING_ENABLED:
        return func
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        if tracer is None:
            return func(*args, **kwargs)
        
        with tracer.start_as_current_span(func.__name__) as span:
            # 记录参数
            span.set_attribute("function.args", str(args)[:500])
            span.set_attribute("function.kwargs", str(kwargs)[:500])
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # 记录返回值摘要
                if isinstance(result, dict):
                    span.set_attribute("function.result_keys", str(list(result.keys())))
                elif isinstance(result, list):
                    span.set_attribute("function.result_length", len(result))
                elif isinstance(result, str):
                    span.set_attribute("function.result_length", len(result))
                
                span.set_attribute("function.success", True)
                
                return result
            except Exception as e:
                span.set_attribute("function.success", False)
                span.set_attribute("function.error", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("function.duration_ms", int(duration * 1000))
    
    return wrapper


class RAGTracing:
    """RAG 操作追踪器"""
    
    def __init__(self, session_id: str = "unknown"):
        self.session_id = session_id
        self.tracer = get_tracer("rag")
    
    def trace_search(self, query: str, top_k: int = 5):
        """追踪一次检索"""
        if self.tracer is None:
            yield None
            return
        
        with self.tracer.start_as_current_span("rag.search") as span:
            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)
            span.set_attribute("session_id", self.session_id)
            yield span
    
    def trace_rerank(self, query: str, num_passages: int):
        """追踪重排序"""
        if self.tracer is None:
            yield None
            return
        
        with self.tracer.start_as_current_span("rag.rerank") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_passages", num_passages)
            yield span
    
    def trace_deepseek(self, model: str, prompt_length: int):
        """追踪 DeepSeek 调用"""
        if self.tracer is None:
            yield None
            return
        
        with self.tracer.start_as_current_span("llm.deepseek") as span:
            span.set_attribute("model", model)
            span.set_attribute("prompt_length", prompt_length)
            yield span


# 初始化
if TRACING_ENABLED:
    try:
        init_tracing()
        print("✅ OpenTelemetry tracing initialized")
    except Exception as e:
        print(f"⚠️ Tracing initialization failed: {e}")
else:
    print("ℹ️ Tracing disabled (set TRACING_ENABLED=true to enable)")