# 生产环境部署指南

## 环境准备

```bash
# 1. 启动 Docker 服务
cd deploy
docker compose up -d  # MySQL Redis Neo4j

# 2. 拉取最新代码
git pull origin main

# 3. 安装依赖
cd backend
source .venv/bin/activate
pip install -r requirements.txt

# 4. 创建模型目录并确保 BGE-M3 已下载
mkdir -p app/graphrag/models/bge-m3/
# 如需重新下载:
python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="BAAI/bge-m3", cache_dir="app/graphrag/models/bge-m3")
EOF

# 5. 初始化数据库
python3 scripts/init_db.py
python3 scripts/import_neo4j.py
```

## 索引构建

```bash
# 构建 BGE-M3 索引
python3 -c "
from app.services.vector_rag_bge_m3 import VectorRAG
import pickle

# 加载文档
with open('app/graphrag/data/output/8e8dca41-982b-53a3-b298-7d6a2b0ed7ee/vector_rag/metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

# 构建索引
rag = VectorRAG()
rag.build_index(meta['chunks'])
rag.save_index('app/graphrag/data/output/8e8dca41-982b-53a3-b298-7d6a2b0ed7ee/vector_rag')
print('索引构建完成')
"
```

## 环境变量配置

```bash
# .env 文件
# DeepSeek 配置
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 向量模型
CHAT_SERVICE=deepseek
OLLAMA_BASE_URL=http://localhost:11434

# 追踪
TRACING_ENABLED=true
OTLP_ENDPOINT=http://localhost:4317
```

## 启动服务

```bash
# 方式一：开发模式
python3 run.py --port 9002 --reload

# 方式二：生产模式
uvicorn app.main:app --host 0.0.0.0 --port 9002 --workers 4

# 方式三：Docker (如需)
docker build -t intelligent-customer .
docker run -d -p 9002:9002 --env-file .env intelligent-customer
```

## 验证部署

```bash
# 健康检查
curl http://localhost:9002/health

# 测试查询
curl -X POST http://localhost:9002/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "博士学制几年？"}' | jq

# 检查指标
curl http://localhost:9002/metrics
```

## 常见问题

### 1. 模型加载慢
首次加载 BGE-M3 需要 2-3 秒。已配置模型热加载。

### 2. CUDA 头寸
检查 `torch.cuda.is_available()`。如无 GPU，系统自动降级到 CPU。

### 3. Neo4j 连接失败
```bash
# 检查容器
docker logs csp-neo4j

# 重置密码
docker exec -it csp-neo4j bin/cypher-shell -u neo4j -p neo4j "ALTER USER neo4j CHANGE NOTIFIED PASSWORD 'neo4j'"
```

### 4. Ollama 连接
```bash
# 查看运行的 Ollama 容器
docker ps | grep ollama

# 重新启动
docker restart <ollama-container>
```