# 问道星辰 · 智能客服系统

> 基于 FastAPI + LangGraph + GraphRAG + Neo4j 的企业级智能客服系统。
> 支持 DeepSeek(百炼) / Ollama 双引擎, 覆盖通用对话、深度思考、图谱查询、文件 RAG 等场景。

---

## 目录结构

```
customer-service-platform/
├── backend/                          # 后端服务 (Python / FastAPI)
│   ├── app/
│   │   ├── api/                      # HTTP 路由层 (auth / chat / conversations / uploads / agent)
│   │   ├── core/                     # 核心配置 / 安全 / 日志 / 中间件
│   │   ├── models/                   # SQLAlchemy 数据模型 (User / Conversation / Message)
│   │   ├── schemas/                  # Pydantic 请求/响应校验
│   │   ├── services/                 # 业务服务层 (LLM / RAG / 缓存 / 搜索 / 索引)
│   │   ├── agent/                    # LangGraph Agent 编排 (路由 / 状态 / 提示词 / 图谱子图)
│   │   ├── tools/                    # 工具定义
│   │   ├── prompts/                  # 提示词模板
│   │   ├── graphrag/                 # Microsoft GraphRAG 2.1.0 (vendored)
│   │   │   └── data/                 # 运行时配置 + 索引输出 (settings.yaml / output/)
│   │   └── main.py                   # FastAPI 应用入口 (唯一)
│   ├── scripts/                      # 运维脚本
│   │   ├── init_db.py                # 初始化 MySQL 表
│   │   ├── import_neo4j.py           # 导入 Northwind 电商数据到 Neo4j
│   │   └── check_env.py              # 环境自检 (一键 smoke)
│   ├── static/dist/                  # Vue3 前端构建产物
│   ├── uploads/                      # 用户上传文件
│   ├── archive/                      # 遗留代码归档
│   ├── .env                          # 环境变量 (已配置)
│   ├── requirements.txt
│   └── run.py                        # 启动入口
├── deploy/
│   └── docker-compose.yml            # MySQL / Redis / Neo4j 容器编排
└── README.md                         # 本文件
```

---

## 快速启动

### 1. 启动基础设施 (Docker)

```bash
cd deploy
docker compose up -d
# MySQL :3307, Redis :6381, Neo4j :7474/:7687
```

### 2. 初始化数据库

```bash
cd backend
./.venv/bin/python scripts/init_db.py       # 创建 MySQL 表
./.venv/bin/python scripts/import_neo4j.py  # 导入 Northwind 数据到 Neo4j
```

### 3. 启动后端服务

```bash
cd backend
./.venv/bin/python run.py
# 默认监听 http://0.0.0.0:9002
# 可选: --port 9002 --no-reload
```

### 4. 环境自检

```bash
./.venv/bin/python scripts/check_env.py
# 一键检查: import / MySQL / Redis / Neo4j / Ollama / DeepSeek / GraphRAG 索引
```

---

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/token` | POST | 登录获取 JWT |
| `/api/auth/users/me` | GET | 当前用户信息 |
| `/api/chat` | POST | 通用对话 (SSE 流式) |
| `/api/reason` | POST | 深度思考 (SSE 流式) |
| `/api/search` | POST | 联网搜索 (SSE 流式) |
| `/api/conversations` | POST | 创建会话 |
| `/api/conversations/user/{id}` | GET | 用户会话列表 |
| `/api/conversations/{id}/messages` | GET | 会话消息 |
| `/api/upload` | POST | 文件上传 + GraphRAG 索引 |
| `/api/upload/image` | POST | 图片上传 |
| `/api/langgraph/query` | POST | LangGraph Agent 查询 (SSE) |
| `/api/langgraph/resume` | POST | 继续中断的 Agent 流程 |

完整 OpenAPI 文档: http://localhost:9002/docs

---

## 模型配置

### 大模型引擎

| 变量 | 值 | 说明 |
|------|-----|------|
| `CHAT_SERVICE` | `deepseek` | 通用对话引擎 |
| `REASON_SERVICE` | `deepseek` | 深度思考引擎 |
| `AGENT_SERVICE` | `deepseek` | LangGraph Agent 引擎 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash-0731` | 百炼 DeepSeek 模型 |
| `VISION_MODEL` | `qwen3-vl-plus` | 图片分析模型 |

### 向量模型

| 用途 | 模型 | 说明 |
|------|------|------|
| Redis 语义缓存 | `nomic-embed-text` (Ollama, 768维) | 对话缓存相似度匹配 |
| GraphRAG 索引 | `qwen3.7-text-embedding` (百炼) | 文件/知识库向量化 |

> **注意**: 两套向量模型独立运行, 维度不同, 互不干扰。**无需本地部署 bge**。

### 切换至 Ollama 本地模型

修改 `.env`:
```env
CHAT_SERVICE=ollama
REASON_SERVICE=ollama
AGENT_SERVICE=ollama
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_REASON_MODEL=deepseek-r1:7b
OLLAMA_AGENT_MODEL=qwen2.5:7b
```

---

## GraphRAG 说明

- **运行时配置**: `app/graphrag/data/settings.yaml` (DashScope 百炼兼容模式)
- **外层 `app/graphrag/settings.yaml`** 为样例模板, **不可直接用于运行时**
- 索引输出: `app/graphrag/data/output/` (entities / communities / relationships 等 parquet 文件)
- 上传文件时自动触发 `IndexingService.process_file()` 构建/更新索引

---

## 端口约定

| 端口 | 服务 | 说明 |
|------|------|------|
| 3307 | MySQL | 映射到容器 3306 |
| 6381 | Redis | 映射到容器 6379 |
| 7474 | Neo4j HTTP | 管理界面 |
| 7687 | Neo4j Bolt | 驱动连接 |
| 9002 | 本后端 | FastAPI 服务 |

> 注意: 本机 8000 被其他项目占用, 本服务统一使用 9002。

---

## 原始章节源码

章节式源码位于 `Agent大型项目实战2：智能客服/` 目录, 为只读归档, 供学习参考。
企业级重构版以本目录 (`customer-service-platform/`) 为准。