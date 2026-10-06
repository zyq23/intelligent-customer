#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境自检脚本: 一键 smoke 检查智能客服后端是否就绪。

用法:
    cd customer-service-platform/backend
    ./.venv/bin/python scripts/check_env.py

检查项:
  1. 关键模块能否 import (app.main / 各 services / LangGraph graph)
  2. MySQL / Redis / Neo4j / Ollama 连通性
  3. DeepSeek(百炼) 模型能否调用
  4. GraphRAG 运行时索引文件是否存在 (data/output/*.parquet)
  5. 服务是否已在默认端口(9002)上提供 /health
"""
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings  # noqa: E402
from app.core.logger import get_logger  # noqa: E402

logger = get_logger(service="check_env")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str):
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def fail(msg: str):
    print(f"  {RED}✗ {msg}{RESET}")


def check_imports() -> list[str]:
    """Step 1: 关键模块 import 自检。返回失败项。"""
    print("1) 关键模块 import")
    errors = []
    modules = [
        "app.main",
        "app.core.config",
        "app.services.llm_factory",
        "app.services.deepseek_service",
        "app.services.redis_semantic_cache",
        "app.services.indexing_service",
        "app.agent.lg_builder",
    ]
    for mod in modules:
        try:
            __import__(mod)
            ok(f"import {mod}")
        except Exception as e:  # noqa: BLE001
            fail(f"import {mod}: {e}")
            errors.append(mod)
    return errors


async def check_infra() -> list[str]:
    """Step 2: 数据库/中间件连通性。返回失败项。"""
    print("2) 基础设施连通性")
    errors = []

    # MySQL
    try:
        import aiomysql
        conn = await aiomysql.connect(
            host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
            password=settings.DB_PASSWORD, db=settings.DB_NAME, connect_timeout=5,
        )
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            await cur.fetchone()
        conn.close()
        ok(f"MySQL {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    except Exception as e:  # noqa: BLE001
        fail(f"MySQL: {e}")
        errors.append("MySQL")

    # Redis
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_timeout=5)
        if r.ping():
            ok(f"Redis {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        else:
            raise ConnectionError("ping failed")
    except Exception as e:  # noqa: BLE001
        fail(f"Redis: {e}")
        errors.append("Redis")

    # Neo4j
    try:
        from neo4j import GraphDatabase
        with GraphDatabase.driver(
            settings.NEO4J_URL, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        ) as driver:
            driver.verify_connectivity()
            ok(f"Neo4j {settings.NEO4J_URL}")
    except Exception as e:  # noqa: BLE001
        fail(f"Neo4j: {e}")
        errors.append("Neo4j")

    # Ollama (若配置使用 ollama 服务才检查)
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5) as resp:
                if resp.status == 200:
                    ok(f"Ollama {settings.OLLAMA_BASE_URL}")
                else:
                    raise ConnectionError(f"HTTP {resp.status}")
    except Exception as e:  # noqa: BLE001
        warn(f"Ollama {settings.OLLAMA_BASE_URL} 不可达: {e} (若只用百炼可忽略)")
    return errors


def check_graphrag_index() -> list[str]:
    """Step 3: GraphRAG 运行时索引文件检查。"""
    print("3) GraphRAG 索引文件")
    data_dir = Path(settings.GRAPHRAG_PROJECT_DIR) / settings.GRAPHRAG_DATA_DIR
    missing = []
    for table in ["entities.parquet", "text_units.parquet", "communities.parquet", "community_reports.parquet"]:
        f = data_dir / "output" / table
        if f.exists():
            ok(f"找到 {f.name}")
        else:
            warn(f"缺少 {table} (未跑过 graphrag build_index, 上传文件时自动构建)")
            missing.append(table)
    return missing


async def check_model() -> list[str]:
    """Step 4: DeepSeek(百炼) 对话/嵌入模型调用。"""
    print("4) 大模型调用")
    errors = []
    try:
        from app.services.model_factory import create_agent_llm
        model = create_agent_llm(temperature=0.2)
        resp = await model.ainvoke([{"role": "user", "content": "回复'模型正常'四个字即可"}])
        ok(f"DeepSeek({settings.DEEPSEEK_MODEL}) 对话: {str(resp.content)[:40]}")
    except Exception as e:  # noqa: BLE001
        fail(f"DeepSeek 对话: {e}")
        errors.append("DeepSeek")
    return errors


async def check_health() -> list[str]:
    """Step 5: 运行中服务的 /health。"""
    print("5) 运行中服务 /health")
    errors = []
    try:
        import aiohttp
        url = "http://localhost:9002/health"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    ok(f"GET {url} -> {await resp.text()}")
                else:
                    raise ConnectionError(f"HTTP {resp.status}")
    except Exception as e:  # noqa: BLE001
        warn(f"{url} 不可达: {e} (服务未启动时可忽略, 运行 `./.venv/bin/python run.py` 启动)")
    return errors


async def main():
    print("=" * 60)
    print("智能客服后端 · 环境自检")
    print("=" * 60)
    total_errors: list[str] = []

    total_errors += check_imports()
    total_errors += await check_infra()
    total_errors += check_graphrag_index()
    total_errors += await check_model()
    total_errors += await check_health()

    print("=" * 60)
    if total_errors:
        print(f"{RED}发现 {len(total_errors)} 项错误:{RESET}")
        for e in set(total_errors):
            print(f"  {RED}- {e}{RESET}")
        print("请根据上方对应 ✗ 项的报错排查后重试。")
        return 1
    print(f"{GREEN}环境检查全部通过 ✓{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
