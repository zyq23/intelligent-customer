# 问道星辰 · 智能客服系统 (intelligent-customer)

基于 **FastAPI + LangGraph + Microsoft GraphRAG + Neo4j + MySQL + Redis** 的企业级智能客服平台。
支持 **DeepSeek（阿里云百炼）** 与 **本地 Ollama** 双引擎，覆盖通用对话、深度思考、图谱知识查询、
文件 RAG 检索、多轮会话、语义缓存等核心场景。

> 仓库结构：核心代码位于 [`customer-service-platform/`](./customer-service-platform) 子目录。

---

## ✨ 核心能力

| 能力 | 说明 |
|------|------|
| 🤖 通用对话 | 基于 LLM 的流式 (SSE) 对话服务，支持深度思考模式 |
| 🔍 图谱知识问答 | Neo4j + Cypher 查询，LangGraph Agent 编排子图 |
| 📄 文件 RAG | 上传文档 → 自动索引 (Microsoft GraphRAG) → 语义/图谱检索 |
| 🧠 向量语义缓存 | Redis + Ollama Embedding，高相似对话直接复用结果 |
| 🌐 联网搜索 | SerpAPI 集成，可配置开启/关闭 |
| 👤 多用户隔离 | 注册/登录 (JWT) + 会话隔离 |
| 📊 数据飞轮 | 对话反馈自动入图，持续优化知识库 |

---

## 🏗 技术栈

```
┌─────────────────────────────────────────────────────────┐
│                 前端 (Vue 3 SPA, static/dist)           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            FastAPI 后端 (customer-service-platform)       │
│  ├─ API 路由: auth / chat / reason / search /            │
│  │             conversations / uploads / langgraph        │
│  ├─ LangGraph Agent (kg_sub_graph: 图谱 + Agentic RAG)  │
│  ├─ Microsoft GraphRAG (vendored, 2.1.0)                │
│  └─ Services: LLM Factory / RAG / Vector Cache / Search  │
└──────────────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │  MySQL   │      │  Neo4j   │      │  Redis   │
  │ (用户/会话)│      │ (知识图谱)│      │ (缓存)   │
  └──────────┘      └──────────┘      └──────────┘
        +
  ┌──────────────────────────────┐
  │  LLM: DeepSeek(百炼) / Ollama │
  └──────────────────────────────┘
```

---

## 📦 目录结构

```
intelligent-customer/
├── customer-service-platform/      # 项目主体
│   ├── backend/                    # Python 后端 (FastAPI)
│   │   ├── app/
│   │   │   ├── api/               # HTTP 路由 (auth / chat / uploads / agent...)
│   │   │   ├── core/              # 配置 / 安全 / 日志 / 中间件
│   │   │   ├── models/            # SQLAlchemy ORM
│   │   │   ├── schemas/           # Pydantic 校验
│   │   │   ├── services/          # LLM / RAG / 缓存 / 搜索 / 索引
│   │   │   ├── agent/             # LangGraph Agent 编排 (子图 / 工具 / 提示词)
│   │   │   └── graphrag/          # Microsoft GraphRAG (vendored)
│   │   ├── scripts/               # init_db / import_neo4j / check_env
│   │   ├── requirements.txt
│   │   └── run.py
│   ├── deploy/
│   │   └── docker-compose.yml     # MySQL / Redis / Neo4j
│   ├── start.sh                   # 一键启动脚本
│   ├── .env.example               # 环境变量模板
│   └── README.md                  # 详细技术文档 (架构 / API / 部署)
├── .gitignore
└── README.md                      # 本文件
```

---

## 🚀 快速开始

```bash
# 1. 克隆并进入项目
git clone https://github.com/zyq23/intelligent-customer
cd intelligent-customer/customer-service-platform

# 2. 配置环境变量
cp .env.example .env
cp backend/.env.example backend/.env
# 编辑 .env 填入你的 API Key 与数据库密码

# 3. 启动基础设施 (MySQL / Redis / Neo4j)
docker compose -f deploy/docker-compose.yml up -d

# 4. 创建 Python 环境并安装依赖
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 5. 初始化数据库与图谱
python scripts/init_db.py
python scripts/import_neo4j.py

# 6. 环境自检
python scripts/check_env.py

# 7. 启动后端服务 (默认 http://0.0.0.0:9002)
python run.py
```

完整文档：[customer-service-platform/README.md](./customer-service-platform/README.md)

---

## 🔌 API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/token` | POST | 登录获取 JWT |
| `/api/auth/users/me` | GET | 当前用户信息 |
| `/api/chat` | POST | 通用对话（SSE 流式） |
| `/api/reason` | POST | 深度思考（SSE 流式） |
| `/api/search` | POST | 联网搜索（SSE 流式） |
| `/api/conversations` | POST | 创建会话 |
| `/api/conversations/{id}/messages` | GET | 会话消息 |
| `/api/upload` | POST | 文件上传 + GraphRAG 索引 |
| `/api/langgraph/query` | POST | LangGraph Agent 查询（SSE） |
| `/api/langgraph/resume` | POST | 恢复中断的 Agent 流程 |

完整 OpenAPI 文档：<http://localhost:9002/docs>

---

## 🛠 端口约定

| 端口 | 服务 | 说明 |
|------|------|------|
| 3307 | MySQL | 映射到容器 3306 |
| 6381 | Redis | 映射到容器 6379 |
| 7474 | Neo4j HTTP | 管理界面 |
| 7687 | Neo4j Bolt | 驱动连接 |
| 9002 | FastAPI 后端 | 主服务 |

> 注意：本机 8000 被其他项目占用，本服务统一使用 **9002**。

---

## 🔐 安全

- 所有密钥均通过 `.env` 管理，仓库中仅保留 `.env.example` 模板
- 请勿将真实 API Key、数据库密码、JWT 密钥提交到版本控制
- 生产环境请更换默认 `SECRET_KEY` 并启用 HTTPS

---

## 📚 相关文档

- [启动指南](./customer-service-platform/STARTUP_GUIDE.md)
- [架构说明](./customer-service-platform/backend/PROJECT_ARCHITECTURE.md)
- [RAG 实现报告](./customer-service-platform/backend/RAG_IMPLEMENTATION_REPORT.md)
- [审计报告](./customer-service-platform/AUDIT_REPORT.md)

---

## 📄 License

MIT © 2025