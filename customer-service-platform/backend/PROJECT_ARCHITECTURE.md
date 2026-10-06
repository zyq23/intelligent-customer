# 问道星辰 · 智能客服系统 — 项目架构详解

> 基于 **FastAPI + LangGraph + Microsoft GraphRAG + 混合向量检索** 的企业级智能客服系统。
> 支持 **DeepSeek / Ollama / OpenAI 兼容接口** 双引擎，覆盖通用对话、深度思考、知识图谱查询、文件 RAG、联网搜索等场景。

> 说明：本文档结合当前代码库实际状态撰写。标注 **[已实现]** 的模块可直接运行；标注 **[部分实现]** 的模块核心逻辑就绪、依赖配置（如向量库/模型切换）后启用；标注 **[规划中]** 的模块为路线图项。这样能保证描述与工程实际一致，而非"纸面架构"。

---

## 一、总体定位与消息处理流水线

整套系统可以理解为一条 **"消息进入 → 意图/检索处理 → 生成回答 → 结束并留痕"** 的流水线。用户输入一句自然语言问题后，系统并不是"一股脑丢给大模型"，而是先做语义预处理，再按问题类型走不同处理路径，最后由主 Agent 统一生成面向用户的回复。

### 消息处理流程

1. **用户问题进入**
   前端（Vue3 构建产物，`static/dist`）以 SSE 流式方式发起请求，用户问题可能口语化、表达不完整、或依赖上下文。

2. **接入与鉴权**
   `app/api/auth.py` 提供注册、登录（JWT）、当前用户信息接口。前端传入 `user_id` 即完成接入，重点落在 AI 业务处理层。

3. **AI 第一次处理（对话编排）**
   主要完成两类工作：
   - **指代消解 / 上下文补全**：结合最近几轮对话，把"那它能退吗"这类不完整表达补全为独立问题（会话历史经 Redis 语义缓存 `redis_semantic_cache.py` 维护）。
   - **意图识别与分流**：判断问题属于"通用对话 / 深度思考 / 联网搜索 / 知识检索 / 图谱查询"等哪一类，进入对应端点。

4. **按意图分流（五个处理出口）**
   - `/api/chat` — 通用对话（SSE 流式）
   - `/api/reason` — 深度思考（SSE 流式）
   - `/api/search` — 联网搜索（SSE 流式）
   - `/api/upload` — 文件上传 + GraphRAG 索引构建
   - `/api/langgraph/query`、`/resume` — LangGraph Agent 图谱/知识查询（支持中断与恢复）

   无法可靠回答时，返回兜底话术或提示转人工。

5. **SSE 流式回答**
   回答通过 SSE 逐字推送，缩短首字等待时间。

6. **结束与留痕**
   会话与消息落库（MySQL），完整处理过程可追溯，为后续问题池 / 数据闭环（规划中）提供基础。

### 架构图核心表达

- 用编号节点展示消息流转；
- 用不同分支体现意图分流、工具调用、图谱检索、拒答与转人工；
- 所有路径最终汇合到统一的主 Agent（LangGraph 编排），由其生成最终回复。

---

## 二、Workflow 与 Agent 的混合架构 **[部分实现]**

系统不把全部决策都交给 Agent，而是采用 **"Workflow + Agent"** 协作模式：

- **Workflow**：负责规则明确、流程固定、需要严格约束的任务，避免模型随意承诺。
- **Agent**：负责需要理解、推理、判断与工具调用的开放性任务。
- **LangGraph**：负责连接流程、维护状态图（节点、分支、循环、中断与恢复）。

### 已有的多 Agent 子图

`app/agent/kg_sub_graph/agentic_rag_agents/workflows/multi_agent/multi_tool.py` 中已构建了一个知识图谱子图，包含以下节点：

```
guardrails → planner → {cypher_query | predefined_cypher | customer_tools}
              ↓
         tool_selection → summarize → final_answer
```

- `guardrails`：输入护栏，拦截不合规请求；
- `planner`：任务规划；
- `cypher_query` / `predefined_cypher`：把自然语言转成 Cypher 查询 / 使用预定义图谱查询；
- `customer_tools`：客服业务工具调用；
- `summarize` → `final_answer`：汇总并生成最终答案。

设计理念与参考项目一致：

> **确定的事情交给流程，需要判断的事情交给 Agent。**

`/api/langgraph/resume` 支持流程中断与恢复 —— 当缺少必要信息时，流程可暂停并等待用户补充后从中断处继续。

---

## 三、系统分层结构

### 1. 交互层 **[已实现]**

- Vue3 前端构建产物（`static/dist/index.html`，约 58KB 单页应用）
- SSE 流式响应
- 职责：客服对话页面、会话展示、流式接收回答

界面刻意保持简洁，把精力集中在后端与 AI 能力。

### 2. 服务层（API 层） **[已实现]**

- 技术：FastAPI、SQLAlchemy（aiomysql 异步驱动）、Pydantic
- 职责：
  - 请求接收与参数校验
  - 用户身份与数据归属校验（`user/{user_id}` 资源隔离）
  - 流式响应（SSE）
  - 连接业务服务、数据库与 AI 编排

主要路由文件：`app/api/{auth,chat,conversations,uploads,agent}.py`。

### 3. 编排层 **[部分实现]**

- 核心：LangGraph
- 职责：
  - 构建状态图，管理意图分流
  - 调度 Agent 与 ReAct / 多 Agent 循环
  - 保存会话状态、支持多轮续接
  - 支持流程中断与恢复（`/resume`）

这是系统的流程控制中心，让系统既有 Workflow 的确定性，又保留 Agent 的灵活性。

### 4. 检索层 **[已实现 + 部分实现]**

这是近期重点优化的层，具备两条可切换的检索路径：

**(a) 简化向量 RAG（FAISS，已实现并跑通）**
- `app/graphrag/rag_query.py` → `SimpleVectorRAG`
- `app/services/vector_rag.py` → `OllamaEmbedder` + `FAISSIndex`
- 用 Ollama embedding + FAISS `IndexFlatIP`（内积，向量已 L2 归一化）做语义检索
- 数据流：`原始 TXT → 文本分块（500 字符/块，50 重叠）→ embedding → FAISS 索引`

**(b) 高级混合 RAG（BGE-M3，已实现核心）**
- `app/graphrag/advanced_rag.py` → `AdvancedRAG`
- **Embedding**：本地 BGE-M3（2.32 GB，GPU/CUDA 运行，1024 维）
- **混合检索**：向量 + BM25（`rank-bm25` 中文分词）双路召回，加权融合
- **重排序**：BGE-Reranker（可选，模型文件已就位）
- **查询扩展**：通过 DeepSeek 做同义问题扩展（依赖 API Key）

**(c) GraphRAG（Microsoft）图谱检索 [部分实现]**
- `app/services/search_service.py` → 加载 GraphRAG 产物（entities / communities / relationships / text_units）
- 向量库后端为 **LanceDB**（`graphrag/vector_stores/lancedb.py`），GraphRAG 实体/关系抽取依赖 Ollama，此前因 embedding 未正确注入而失败，已通过 BGE-M3 / Ollama 方案补全索引。

### 5. 存储层 **[已实现]**

- **MySQL**（`mysql+aiomysql`，端口 3307）：
  - 用户 / 会话 / 消息
  - 上传文件与 GraphRAG 索引产物
  - 审计与会话留痕
- **Redis**（端口 6381）：
  - 语义缓存（`redis_semantic_cache.py`，按 `user_id` 隔离）
  - 高频问答命中缓存，降低 LLM 成本
- **Neo4j**（规划中启用）：
  - 知识图谱存储，供 `cypher_query` 节点查询使用

### 6. 支撑系统 **[规划中]**

- **全链路追踪（Langfuse / LangSmith）**：记录 LLM、工具、工作流调用链，支持链路排查 —— 当前尚未接入
- **问题池 / 数据飞轮**：拒答、自评失败、用户负反馈问题进入问题池，人工审核补回知识库 —— 当前留痕能力已有，闭环待建设
- **意图分类器（可选微调）**：先判断 Prompt + RAG 是否足以解决，必要时再引入轻量分类器

---

## 四、关键技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 语言 | Python 3.12 | 后端主语言 |
| Web 框架 | FastAPI + uvicorn | 异步高性能 API，默认端口 9002/9003 |
| ORM | SQLAlchemy (aiomysql) | 异步访问 MySQL |
| 编排 | LangGraph | 状态图、多 Agent、中断恢复 |
| 模型接入 | LangChain | 模型封装、Prompt、工具调用 |
| 向量库 | FAISS（简化 RAG）/ LanceDB（GraphRAG） | 语义检索 |
| Embedding | BGE-M3（本地 1024 维）/ Ollama | 中文语义编码 |
| 重排序 | BGE-Reranker（可选） | 召回结果精排 |
| 混合检索 | BM25（rank-bm25） | 关键词召回 |
| 图谱 | Neo4j + Microsoft GraphRAG | 实体关系抽取与 Cypher 查询 |
| 缓存 | Redis 语义缓存 | 高频问答命中 |
| 大模型 | DeepSeek（主力）/ Ollama / OpenAI 兼容 | 双引擎可切换 |
| 部署 | Docker Compose | MySQL/Redis/Neo4j 基础设施 |
| 前端 | Vue3 + SSE | 构建产物随后端静态托管 |

---

## 五、核心设计理念

1. **规则与推理分离**
   有明确业务规则的场景（如订单、退款类高风险操作）走确定性 Workflow；意图理解、复杂问答、工具选择交给 Agent。

2. **业务边界优先**
   不止追求回答流畅，还要求回答符合业务约束。证据不足时**优先拒答或转人工**，而非让模型自行补全事实。

3. **证据驱动回答**
   RAG 不只是生成答案，还为回答提供可引用的依据（检索快照 + 来源文件），增强可解释性与可靠性。

4. **流程可暂停、可恢复**
   缺少关键信息（如订单号）时不猜测，暂停流程，用户补充后从中断处继续（`/api/langgraph/resume`）。

5. **先解决检索，再考虑微调**
   优先用 Prompt + RAG 解决，只有检索/提示词确实无能为力时，才引入轻量分类器微调。

---

## 六、数据飞轮（规划中）

数据飞轮是架构中较具特色的一环，入口包括三类：

1. 检索证据不足而被拒答的问题；
2. 模型生成前自评未通过的问题；
3. 用户主动点踩的问题。

这些问题连同当时的检索快照进入问题池，由人工按提问频次审核，判断是"知识库缺答案 / 有知识没检索到 / 检索或生成有问题"。审核后补充答案并重新向量化入库，使后续相似问题能被正确回答。

构成闭环：

**用户问题 → 检索与回答 → 失败记录 → 人工审核 → 知识补充 → 向量化入库 → 后续可回答。**

> 当前系统已具备会话留痕与审计能力，飞轮闭环是下一步建设重点。

---

## 七、当前落地实用性评估（实事求是）

| 模块 | 状态 | 说明 |
|------|------|------|
| 通用/深度/搜索对话 API | ✅ 已实现 | `/chat` `/reason` `/search` SSE 流式 |
| 用户体系（注册/登录/JWT） | ✅ 已实现 | `auth.py` |
| 会话与消息管理 | ✅ 已实现 | `conversations.py`，MySQL 落库 |
| 文件上传 + 解析 | ✅ 已实现 | `uploads.py`，支持文本抽取 |
| 简化向量 RAG（FAISS） | ✅ 已跑通 | 893 文档块，实测命中良好 |
| 高级混合 RAG（BGE-M3+BM25） | ✅ 核心就绪 | 需切换 LangChain embedder + 重建索引后即可用于服务 |
| Redis 语义缓存 | ✅ 已实现 | 高频问答降本 |
| LangGraph Agent 子图 | 🔶 部分实现 | 节点就绪，缺 OllamaEmbedder 注入与端到端验证 |
| GraphRAG 图谱 | 🔶 部分实现 | 索引产物已生成，Neo4j 启用为规划项 |
| 全链路追踪 / 数据飞轮 | ⬜ 规划中 | 留痕已有，闭环待建设 |

**结论**：核心的"对话 + 检索 + 缓存 + 会话留痕"链路已经可用；GraphRAG / LangGraph 子图属于增强能力，处于"配置后即可启用"状态。生产化落地的关键动作是：**把 BGE-M3 作为 LangChain embedder 注入服务层，重建一次完整索引，再补上 Neo4j 与全链路追踪**。

---

## 八、快速启动

```bash
# 1. 基础设施
cd deploy && docker compose up -d      # MySQL:3307  Redis:6381  Neo4j:7474

# 2. 初始化
cd backend
./.venv/bin/python scripts/init_db.py
./.venv/bin/python scripts/check_env.py   # 一键自检

# 3. 启动后端
./.venv/bin/python run.py                 # http://0.0.0.0:9002
```

OpenAPI 文档：`http://localhost:9002/docs`
