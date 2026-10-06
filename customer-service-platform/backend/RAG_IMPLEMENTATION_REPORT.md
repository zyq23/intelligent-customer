# 简化版向量 RAG 系统实施报告

## 📋 概述

成功实现了基于 **Ollama nomic-embed-text** + **FAISS** 的简化版向量 RAG 系统，解决了 GraphRAG 实体提取时的超时问题。

## ✅ 已完成任务

### 1. Ollama Embedding 配置
- **问题**: bge-m3 模型未下载，本地 Ollama 端口冲突
- **解决方案**: 
  - 使用 Docker 容器的 Ollama 服务 (端口 21434)
  - 拉取 `nomic-embed-text` 模型作为替代 (274MB，比 bge-m3 轻量)
  - 确认 embedding 维度：768

### 2. 向量 RAG 实现
- **位置**: `/data/zyq/intelligent-customer/customer-service-platform/backend/app/graphrag/`
- **主要组件**:
  - `rag_query.py`: 查询接口模块
  - `app/graphrag/data/output/8e8dca41-982b-53a3-b298-7d6a2b0ed7ee/vector_rag/`: 索引存储目录
    - `faiss.index`: FAISS 索引文件
    - `embeddings.npy`: 文档向量数组
    - `metadata.pkl`: 文档分片和来源元数据

### 3. 数据处理流程
```
原始 TXT 文档 → 文本分块 (500 字符/块，重叠 50 字符) → Ollama embedding → FAISS 索引
```
- **处理文档数**: 用户 `8e8dca41-982b-53a3-b298-7d6a2b0ed7ee` 的所有 TXT 文件
- **生成文档块**: 893 个
- **索引维度**: 768

## 🔍 查询测试

### 测试问题与结果示例

#### 问题 1: "硕士毕业需要什么条件?"
- **Top 1**: 相似度 271.914 - 培养目标、学制描述
- **Top 2**: 相似度 262.930 - 专业实践要求

#### 问题 2: "如何申请论文答辩?"
- **Top 1**: 相似度 263.510 - 学术规范相关内容
- **Top 2**: 相似度 262.589 - 阅读指导内容

#### 问题 3: "研究生毕业需要什么科研成果?"
- **Top 1**: 相似度 258.318 - 课程内容
- **Top 2**: 相似度 252.659 - 教学目的及要求
- **Top 3**: 相似度 251.761 - VR 相关研究论文要求

### 性能指标
- **单次 embedding 耗时**: ~2-5 秒 (取决于 GPU 负载)
- **893 个文档块编码**: 约 75 分钟 (串行处理)
- **FAISS 索引构建**: <1 秒
- **查询响应**: <5 秒 (含 embedding)

## 📁 文件清单

```
backend/
├── app/graphrag/
│   ├── rag_query.py              # 查询接口 (新建)
│   └── data/
│       ├── input/
│       │   └── 8e8dca41-982b-53a3-b298-7d6a2b0ed7ee/
│       │       └── *.txt         # 源文档
│       └── output/
│           └── 8e8dca41-982b-53a3-b298-7d6a2b0ed7ee/
│               └── vector_rag/
│                   ├── faiss.index
│                   ├── embeddings.npy
│                   └── metadata.pkl
├── .env                          # 环境变量
└── RAG_IMPLEMENTATION_REPORT.md  # 本文档
```

## 🛠️ 使用方法

### 1. 命令行查询
```bash
cd backend
source .venv/bin/activate
python app/graphrag/rag_query.py "你的问题"
```

### 2. Python API
```python
from app.graphrag.rag_query import SimpleVectorRAG

rag = SimpleVectorRAG("8e8dca41-982b-53a3-b298-7d6a2b0ed7ee")
results = rag.query("你的问题", top_k=5)

for r in results:
    print(f"[{r['rank']}] {r['similarity_score']:.3f}: {r['content'][:100]}...")
```

### 3. 环境变量配置
```bash
# 修改 OLLAMA_BASE_URL 指向不同的 Ollama 实例
export OLLAMA_BASE_URL="http://your-ollama-server:21434"
```

## ⚙️ 系统依赖

- **Ollama**: Docker 容器运行，端口 21434
- **Embedding 模型**: `nomic-embed-text:latest`
- **Python 包**: `faiss-cpu`, `numpy`, `requests`, `sentence-transformers`
- **GPU**: 未使用 (nomic-embed-text 在 CPU 上运行)

## 🔄 后续优化建议

1. **批处理优化**: 将串行 embedding 改为并发请求
2. **GPU 加速**: 配置 Ollama 使用 GPU 进行 embedding 推理
3. **混合搜索**: 结合 BM25 关键字搜索和向量搜索
4. **重排序**: 添加交叉编码器对 Top-K 结果重排序
5. **缓存策略**: 缓存热门问题的 embedding 结果
6. **分块优化**: 根据文档结构智能分块而非固定长度

## ❗ 已知限制

- **HuggingFace 不可达**: 无法下载 bge-m3 到本地，只能用 nomic-embed-text
- **本地 Ollama 不可用**: 端口冲突和进程卡死，使用 Docker 版本
- **编码速度慢**: 串行处理 893 个块耗时较长

## 📊 对比: GraphRAG vs 简化版 RAG

| 特性 | GraphRAG (原始方案) | 简化版 RAG (当前实现) |
|------|---------------------|---------------------|
| 实体提取 | 需要，常超时 | 不需要 |
| 关系图谱 | 需要，复杂 | 不需要 |
| 向量化 | 可选 | 必需 |
| 响应时间 | 几分钟到几十分钟 | 几秒到几十秒 |
| 适用场景 | 复杂知识推理 | 语义检索 |
| 资源消耗 | 高 | 低 |

## 👨‍💻 维护说明

### 重建索引
当有新的 TXT 文档添加到 `app/graphrag/data/input/{user_id}/` 时，重新运行嵌入编码和 FAISS 索引构建。

### 更换 Embedding 模型
修改 `rag_query.py` 中的 `EMBEDDING_MODEL` 变量，并确保对应的 Ollama 模型已加载。

---
**生成日期**: 2026-10-03  
**系统版本**: v1.0  
**作者**: ZCode Agent
