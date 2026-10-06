# 问道星辰 - 智能客服系统启动指南

## 📋 系统概述

这是一个企业级智能客服系统，包含以下组件：

- **后端**: FastAPI + LangGraph + RAG
- **数据库**: MySQL 8.0
- **缓存**: Redis
- **知识图谱**: Neo4j
- **前端**: 现代化单页应用（已内置）

---

## 🚀 快速启动

### 1️⃣ 启动基础设施（Docker 容器）

```bash
cd /data/zyq/intelligent-customer/customer-service-platform

# 查看所有服务状态
docker compose -f deploy/docker-compose.yml ps

# 如果服务未运行，启动它们
docker compose -f deploy/docker-compose.yml up -d
```

**端口映射：**
- MySQL: `3307` (容器内 `3306`)
- Redis: `6381` (容器内 `6379`)
- Neo4j HTTP: `7474`
- Neo4j Bolt: `7687`

### 2️⃣ 初始化数据库

```bash
cd /data/zyq/intelligent-customer/customer-service-platform/backend

# 激活 Python 虚拟环境
source .venv/bin/activate

# 初始化 MySQL 数据库表
python scripts/init_db.py

# 导入 Neo4j 知识图谱数据
python scripts/import_neo4j.py
```

### 3️⃣ 启动后端服务

```bash
# 方式一：前台运行（推荐开发调试）
python run.py --no-reload

# 方式二：后台运行
python run.py --no-reload > server.log 2>&1 &

# 方式三：使用 nohup
nohup python run.py --no-reload > server.log 2>&1 &
```

服务将在 `http://localhost:9003` 上启动。

---

## 🌐 访问系统

### 前端界面
打开浏览器访问：**http://localhost:9003**

功能包括：
- ✅ 用户登录/注册
- ✅ 实时聊天界面
- ✅ 对话历史管理
- ✅ 多轮对话支持

### API 文档
- **Swagger UI**: http://localhost:9003/docs
- **OpenAPI JSON**: http://localhost:9003/openapi.json

---

## 🔧 API 接口说明

### 认证相关

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/register` | 用户注册 |
| POST | `/api/token` | 获取访问令牌 |
| GET | `/api/users/me` | 获取当前用户信息 |

### 对话相关

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/conversations` | 创建新对话 |
| GET | `/api/conversations/user/{user_id}` | 获取用户对话列表 |
| GET | `/api/conversations/{conversation_id}/messages` | 获取对话消息 |
| DELETE | `/api/conversations/{conversation_id}` | 删除对话 |

### 智能客服

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/chat` | 发送消息并获取回复 |
| POST | `/api/reason` | 推理分析 |
| POST | `/api/search` | 知识检索 |

---

## 💡 使用示例

### 用户注册
```bash
curl -X POST http://localhost:9003/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "Test123456"
  }'
```

### 用户登录
```bash
curl -X POST http://localhost:9003/api/token \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test123456"
  }'
```

### 发送消息
```bash
curl -X POST http://localhost:9003/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "conversation_id": 1,
    "message": "你好，请问有什么可以帮助你的？",
    "user_id": 1
  }'
```

---

## 🔍 故障排查

### 问题 1: 无法连接到 MySQL

**症状**: `Can't connect to MySQL server on localhost:3307`

**解决方案**:
```bash
# 检查 Docker 容器状态
docker compose -f deploy/docker-compose.yml ps

# 重启 MySQL 容器
docker compose -f deploy/docker-compose.yml restart csp-mysql
```

### 问题 2: 前端无法显示

**症状**: 浏览器访问 http://localhost:9003 显示空白或错误

**解决方案**:
1. 确认后端服务正在运行：`pgrep -f "run.py"`
2. 检查静态文件是否存在：`ls backend/static/dist/`
3. 查看服务器日志：`tail -f server.log`

### 问题 3: API 返回 500 错误

**症状**: 调用 API 时返回 Internal Server Error

**解决方案**:
1. 查看后端日志中的错误信息
2. 确认数据库已正确初始化：`python scripts/init_db.py`
3. 确认 Redis 连接正常：`redis-cli -p 6381 ping`

### 问题 4: 知识库问答无响应

**症状**: 发送消息后长时间无响应或返回空

**解决方案**:
1. 确认 Neo4j 数据已导入：`python scripts/import_neo4j.py`
2. 检查 Neo4j 连接：`docker exec -it csp-neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n)"`
3. 查看 RAG 配置是否正确

---

## 🛠️ 开发指南

### 项目结构
```
customer-service-platform/
├── backend/                  # 后端代码
│   ├── app/
│   │   ├── api/             # API 路由
│   │   ├── core/            # 核心配置
│   │   ├── models/          # 数据模型
│   │   ├── schemas/         # Pydantic 模式
│   │   ├── services/        # 业务逻辑
│   │   └── agent/           # LangGraph Agent
│   ├── static/dist/         # 前端构建产物
│   └── scripts/             # 工具脚本
├── deploy/                  # Docker 配置
└── docs/                    # 文档
```

### 修改 API 端点

编辑 `backend/app/api/` 下的对应文件，例如 `chat.py`:
```python
@router.post("/chat")
async def chat_request(data: ChatRequest, db: AsyncSession = Depends(get_db)):
    # 实现逻辑
    return {"response": "AI 回复内容"}
```

### 添加新的 API 路由

1. 在 `backend/app/api/` 创建新的路由文件
2. 在 `backend/app/api/__init__.py` 中注册路由
3. 重启后端服务

---

## 📝 注意事项

1. **首次使用时**确保所有 Docker 容器都已启动并且健康
2. **数据库初始化**只需要执行一次，后续无需重复
3. **Neo4j 数据导入**在每次更新知识库后需要重新执行
4. **生产环境部署**请修改 `backend/app/core/config.py` 中的配置

---

## 🆘 获取帮助

遇到问题时：
1. 查看后端日志：`tail -f server.log`
2. 查看 Docker 容器日志：`docker logs csp-backend`
3. 检查 API 文档：http://localhost:9003/docs

---

**祝使用愉快！** 🎉
