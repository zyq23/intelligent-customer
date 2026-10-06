from app.agent.lg_states import AgentState, Router
from app.agent.lg_prompts import (
    ROUTER_SYSTEM_PROMPT,
    GET_ADDITIONAL_SYSTEM_PROMPT,
    GENERAL_QUERY_SYSTEM_PROMPT,
    GET_IMAGE_SYSTEM_PROMPT,
    GUARDRAILS_SYSTEM_PROMPT,
    RAGSEARCH_SYSTEM_PROMPT,
    CHECK_HALLUCINATIONS,
    GENERATE_QUERIES_SYSTEM_PROMPT
)
from langchain_core.runnables import RunnableConfig
from app.core.config import settings, ServiceType
from app.core.logger import get_logger
from app.services.model_factory import create_agent_llm
from typing import cast, Literal, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from app.agent.lg_states import AgentState, InputState, Router, GradeHallucinations
from app.agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.northwind_retriever import NorthwindCypherRetriever
from app.agent.kg_sub_graph.agentic_rag_agents.components.planner.node import create_planner_node
from app.agent.kg_sub_graph.agentic_rag_agents.workflows.multi_agent.multi_tool import create_multi_tool_workflow
from app.agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from pydantic import BaseModel
from typing import Dict, List
from langchain_core.messages import AIMessage
from langchain_core.runnables.base import Runnable
from app.agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import retrieve_and_parse_schema_from_graph_for_prompts
from langchain_core.prompts import ChatPromptTemplate
import base64
import os
import aiohttp
import asyncio
import json
import time
import uuid
from pathlib import Path


from typing import Literal
from pydantic import BaseModel, Field


class AdditionalGuardrailsOutput(BaseModel):
    """
    格式化输出，用于判断用户的问题是否与图谱内容相关
    """
    decision: Literal["end", "continue"] = Field(
        description="Decision on whether the question is related to the graph contents."
    )


# 构建日志记录器
logger = get_logger(service="lg_builder")

async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    # 选择模型实例，通过.env文件中的AGENT_SERVICE参数选择
    model = create_agent_llm(temperature=0.7, tags=["router"])
    logger.info(f"Using agent model: {settings.DEEPSEEK_MODEL if settings.AGENT_SERVICE == ServiceType.DEEPSEEK else settings.OLLAMA_AGENT_MODEL}")

    # 拼接提示模版 + 用户的实时问题（包含历史上下文对话） 
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ] + state.messages
    logger.info("-----Analyze user query type-----")
    logger.info(f"History messages: {state.messages}")
    
    # 使用结构化输出，输出问题类型
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    logger.info(f"Analyze user query type completed, result: {response}")
    return {"router": response}

def route_query(
    state: AgentState,
) -> Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"]:
    """根据查询分类确定下一步操作。

    Args:
        state (AgentState): 当前代理状态，包括路由器的分类。

    Returns:
        Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"]: 下一步操作。
    """
    _type = state.router["type"]
    
    # 检查配置中是否有图片路径，如果有，优先处理为图片查询
    if hasattr(state, "config") and state.config and state.config.get("configurable", {}).get("image_path"):
        logger.info("检测到图片路径，转为图片查询处理")
        return "create_image_query"

    if _type == "general-query":
        return "respond_to_general_query"
    elif _type == "additional-query":
        return "get_additional_info"
    elif _type == "graphrag-query":
        return "create_research_plan"
    elif _type == "image-query":
        return "create_image_query"
    elif _type == "file-query":
        return "create_file_query"
    else:
        raise ValueError(f"Unknown router type {_type}")
    
async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成对一般查询的响应，完全基于大模型，不会触发任何外部服务的调用，包括自定义工具、知识库查询等。

    当路由器将查询分类为一般问题时，将调用此节点。

    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("-----generate general-query response-----")
    
    # 使用大模型生成回复
    model = create_agent_llm(temperature=0.7, tags=["general_query"])
    
    system_prompt = GENERAL_QUERY_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def get_additional_info(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成一个响应，要求用户提供更多信息。

    当路由确定需要从用户那里获取更多信息时，将调用此函数。

    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("------continue to get additional info------")
    
    # 使用大模型生成回复
    model = create_agent_llm(temperature=0.7, tags=["additional_info"])

    # 如果用户的问题是电商相关，但与自己的业务无关，则需要返回"无关问题"

    # 首先连接 Neo4j 图数据库
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")

    # 定义电商经营范围
    scope_description = """
    个人电商经营范围：智能家居产品，包括但不限于：
    - 智能照明（灯泡、灯带、开关）
    - 智能安防（摄像头、门锁、传感器）
    - 智能控制（温控器、遥控器、集线器）
    - 智能音箱（语音助手、音响）
    - 智能厨电（电饭煲、冰箱、洗碗机）
    - 智能清洁（扫地机器人、洗衣机）
    
    不包含：服装、鞋类、体育用品、化妆品、食品等非智能家居产品。
    """

    scope_context = (
        f"参考此范围描述来决策:\n{scope_description}"
        if scope_description is not None
        else ""
    )

    # 动态从 Neo4j 图表中获取图表结构
    graph_context = (
        f"\n参考图表结构来回答:\n{retrieve_and_parse_schema_from_graph_for_prompts(neo4j_graph)}"
        if neo4j_graph is not None
        else ""
    )

    message = scope_context + graph_context + "\nQuestion: {question}"

    # 拼接提示模版
    full_system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                GUARDRAILS_SYSTEM_PROMPT,
            ),
            (
                "human",
                (message),
            ),
        ]
    )

    # 构建格式化输出的 Chain， 如果匹配，返回 continue，否则返回 end
    guardrails_chain = full_system_prompt | model.with_structured_output(AdditionalGuardrailsOutput)
    guardrails_output = await guardrails_chain.ainvoke(
            {"question": state.messages[-1].content if state.messages else ""}
        )

    # 根据格式化输出的结果，返回不同的响应
    if guardrails_output.decision == "end":
        logger.info("-----Fail to pass guardrails check-----")
        return {"messages": [AIMessage(content="抱歉，我家暂时没有这方面的商品，可以在别家看看哦~")]}
    else:
        logger.info("-----Pass guardrails check-----")
        system_prompt = GET_ADDITIONAL_SYSTEM_PROMPT.format(
            logic=state.router["logic"]
        )
        messages = [{"role": "system", "content": system_prompt}] + state.messages
        response = await model.ainvoke(messages)
        return {"messages": [response]}

async def create_image_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """处理图片查询并生成描述回复
    
    Args:
        state (AgentState): 当前代理状态，包括对话历史
        config (RunnableConfig): 配置参数，包含线程ID等配置信息
        
    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应
    """
    logger.info("-----Found User Upload Image-----")    
    image_path = config.get("configurable", {}).get("image_path", None)

    if not image_path or not Path(image_path).exists():
        logger.warning(f"User Upload Image Not Found: {image_path}")
        return {"messages": [AIMessage(content="抱歉，我无法查看这张图片，请重新上传。")]}
    
    # 获取视觉模型配置
    api_key = settings.VISION_API_KEY
    base_url = settings.VISION_BASE_URL
    vision_model = settings.VISION_MODEL
    
    if not api_key or not base_url or not vision_model:
        logger.error("Vision Model Configuration Not Complete")
        return {"messages": [AIMessage(content="抱歉，我无法查看这张图片，请重新上传。")]}
    
    logger.info(f"Using Vision Model: {vision_model} to process image: {image_path}")
    
    try:
        # 导入图片处理库
        from PIL import Image
        import io
        
        # 读取并压缩图片
        with Image.open(image_path) as img:
            # 设置最大尺寸
            max_size = 1024
            # 计算缩放比例
            width, height = img.size
            ratio = min(max_size / width, max_size / height)
            
            # 如果图片尺寸已经小于最大尺寸，不需要缩放
            if width <= max_size and height <= max_size:
                resized_img = img
            else:
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为JPEG格式，并调整质量
            img_byte_arr = io.BytesIO()
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
            resized_img.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr.seek(0)
            
            # 转换为base64
            image_data = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            
            logger.info(f"Image Compressed, Original Size: {width}x{height}, New Size: {resized_img.width}x{resized_img.height}")
        
        # 构建API请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的图像分析助手。请详细分析图片中的内容，特别关注产品细节、品牌、型号等信息。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.7
        }
        
        # 发送API请求
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60  # 增加超时时间
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Successfully processed image and generated description")
                    # 使用图片描述和用户问题生成最终回复
                    # 从lg_prompts导入电商客服模板
                    
                    # 构建回复请求
                    model = create_agent_llm(temperature=0.7, tags=["image_query"])
                    # 使用专门的图片查询提示模板
                    system_prompt = GET_IMAGE_SYSTEM_PROMPT.format(
                        image_description=image_description
                    )
                    messages = [{"role": "system", "content": system_prompt}] + state.messages
                    response = await model.ainvoke(messages)
                    return {"messages": [response]}    
        
                else:
                    error_text = await response.text()
                    logger.error(f"Vision API Request Failed: {response.status} - {error_text}")
                    return {"messages": [AIMessage(content=f"抱歉，我无法查看这张图片，请重新上传。")]}





    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"messages": [AIMessage(content=f"抱歉，我无法查看这张图片，请重新上传。")]}

async def create_file_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """处理用户上传的文件查询, 基于本地 GraphRAG 文件索引回答用户问题.

    路由入口 /api/langgraph/query 会把用户上传的文件保存到 uploads/ 目录,
    并将绝对路径放入 config["configurable"]["file_path"]。本节点:

    1. 读取 file_path, 若缺失/不存在则返回可见的中文提示(绝不静默失败);
    2. 复用 IndexingService.process_file 对该文件构建/更新 GraphRAG 索引,
       索引输出会写到 data/output/<user_uuid>/ (而非顶层的 Northwind 索引);
    3. 用本地 GraphRAG local_search 检索最相关内容(直接加载该用户目录的索引);
    4. 用通用大模型基于检索结果生成面向用户的回复。

    Args:
        state (AgentState): 当前代理状态, 包含对话历史。
        config (RunnableConfig): 线程配置, 含 file_path / user_id 等可配置项。

    Returns:
        Dict[str, List[BaseMessage]]: 包含 'messages' 键的字典, 存放生成的回复。
    """
    logger.info("-----execute file query-----")

    # 1. 获取用户上传的文件路径
    file_path = (config.get("configurable") or {}).get("file_path", None)
    if not file_path or not Path(file_path).exists():
        logger.warning(f"User uploaded file not found or not provided: {file_path}")
        return {
            "messages": [
                AIMessage(
                    content=(
                        "抱歉，我没有收到您的文件～ 🙏 请先通过「上传文件」功能把文档/PDF/图片上传给我，"
                        "再告诉我想了解文件中的什么内容，我会帮您分析~"
                    )
                )
            ]
        }

    logger.info(f"Processing uploaded file for query: {file_path}")

    # 2. 构建/更新该文件的 GraphRAG 索引 (输出写到 data/output/<user_uuid>)
    from app.services.indexing_service import IndexingService
    indexing_service = IndexingService()
    file_name = Path(file_path).name
    # user_id 默认取当前任务所属用户(在 config 中传递), 无则用 0(共享目录)
    user_id = (config.get("configurable") or {}).get("user_id", 0)
    file_info = {
        "filename": file_name,
        "original_name": file_name,
        "path": str(file_path).replace("\\", "/"),
        "user_id": user_id,
    }
    index_result = await indexing_service.process_file(file_info)

    if index_result.get("status") != "success":
        err = index_result.get("error") or index_result.get("errors") or "未知错误"
        logger.error(f"File indexing failed for {file_path}: {err}")
        return {
            "messages": [
                AIMessage(
                    content=f"抱歉，文件 {file_name} 暂时无法解析处理～ 😔 目前已尽力处理但仍未成功（原因：{err}）请换个格式试试（如 .txt / .pdf）。"
                )
            ]
        }

    # 3. 基于该文件(用户目录)的 GraphRAG 索引检索, 回答用户当前问题
    try:
        # 直接使用本项目的 GraphRAG API 对 "该用户专属的索引目录" 做 local_search,
        # 而不是顶层 GraphRAGAPI() 默认里的 Northwind 全局索引。
        import app.graphrag.graphrag.api as api
        from app.graphrag.graphrag.config.load_config import load_config
        from app.graphrag.graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
        from app.graphrag.graphrag.utils.storage import load_table_from_storage
        from app.graphrag.graphrag.storage.file_pipeline_storage import FilePipelineStorage

        user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
        user_output_dir = os.path.join(
            settings.GRAPHRAG_PROJECT_DIR, settings.GRAPHRAG_DATA_DIR,
            "output", user_uuid,
        )
        if not os.path.isdir(user_output_dir):
            raise FileNotFoundError(f"用户索引目录不存在: {user_output_dir}")

        # 配置从含 settings.yaml 的 data/ 根目录加载,
        # 而多组 parquet 索引数据从该用户专属的 output/<user_uuid>/ 目录读取。
        data_root_dir = os.path.join(
            settings.GRAPHRAG_PROJECT_DIR, settings.GRAPHRAG_DATA_DIR
        )
        file_cfg = load_config(Path(data_root_dir), None, None)
        storage = FilePipelineStorage(root_dir=str(user_output_dir))
        entities = await load_table_from_storage("entities", storage)
        text_units = await load_table_from_storage("text_units", storage)
        communities = await load_table_from_storage("communities", storage)
        community_reports = await load_table_from_storage("community_reports", storage)
        relationships = await load_table_from_storage("relationships", storage)
        try:
            covariates = await load_table_from_storage("covariates", storage)
        except Exception:
            covariates = None

        context_data = {}
        callbacks: list = []

        def on_context(context):
            context_data["_ctx"] = context

        cb = NoopQueryCallbacks()
        cb.on_context = on_context
        callbacks.append(cb)

        last_question = state.messages[-1].content if state.messages else ""
        response, _ = await api.local_search(
            config=file_cfg,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates,
            community_level=int(settings.GRAPHRAG_COMMUNITY_LEVEL),
            response_type=settings.GRAPHRAG_RESPONSE_TYPE,
            query=last_question,
            callbacks=callbacks,
        )
        retrieved = response or ""
        logger.info(f"File-local search got {len(str(retrieved))} chars")
    except Exception as e:
        logger.error(f"File RAG retrieval failed: {e}", exc_info=True)
        retrieved = ""

    # 4. 用通用模型基于检索结果生成回答
    model = create_agent_llm(temperature=0.7, tags=["file_query"])
    system_prompt = (
        "你是一个电商领域的智能客服。用户上传了一个文件并基于该文件提问。"
        "请严格基于下面给出的文件内容回答, 如果文件内容不足以回答, 就如实说明, 不要编造。\n\n"
        f"[文件内容]\n{retrieved}\n\n[文件内容结束]"
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[str] | str]:
    """通过查询本地知识库回答客户问题，执行任务分解，创建分布查询计划。

    Args:
        state (AgentState): 当前代理状态，包括对话历史。
        config (RunnableConfig): 用于配置计划生成的模型。

    Returns:
        Dict[str, List[str] | str]: 包含'steps'键的字典，其中包含研究步骤列表。
    """
    logger.info("------execute local knowledge base query------")

    # 使用大模型生成查询/多跳、并行查询计划
    model = create_agent_llm(temperature=0.7, tags=["research_plan"])
    
    # 初始化必要参数
    # 1. Neo4j图数据库连接 - 使用配置中的连接信息
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")

    # 2. 创建自定义检索器实例，根据 Graph Schema 创建 Cypher 示例，用来引导大模型生成正确的Cypher 查询语句
    cypher_retriever = NorthwindCypherRetriever()

    # step 3. 定义工具模式列表    
    from app.agent.kg_sub_graph.kg_tools_list import cypher_query, predefined_cypher, microsoft_graphrag_query
    tool_schemas: List[type[BaseModel]] = [cypher_query, predefined_cypher, microsoft_graphrag_query]

    #from app.agent.kg_sub_graph.kg_tools_list import cypher_query, predefined_cypher, customer_tools
    #tool_schemas: List[type[BaseModel]] = [cypher_query, predefined_cypher, customer_tools]

    # 3. 预定义的Cypher查询 - 为电商场景定义有用的查询
    from app.agent.kg_sub_graph.agentic_rag_agents.components.predefined_cypher.cypher_dict import predefined_cypher_dict

    # 定义电商经营范围
    scope_description = """
    个人电商经营范围：智能家居产品，包括但不限于：
    - 智能照明（灯泡、灯带、开关）
    - 智能安防（摄像头、门锁、传感器）
    - 智能控制（温控器、遥控器、集线器）
    - 智能音箱（语音助手、音响）
    - 智能厨电（电饭煲、冰箱、洗碗机）
    - 智能清洁（扫地机器人、洗衣机）
    
    不包含：服装、鞋类、体育用品、化妆品、食品等非智能家居产品。
    """

    # 创建多工具工作流
    multi_tool_workflow = create_multi_tool_workflow(
        llm=model,
        graph=neo4j_graph,
        tool_schemas=tool_schemas,
        predefined_cypher_dict=predefined_cypher_dict,
        cypher_example_retriever=cypher_retriever,
        scope_description=scope_description,
        llm_cypher_validation=True,
    )
    
    # return multi_tool_workflow
    # 准备输入状态
    last_message = state.messages[-1].content if state.messages else ""
    input_state = {
        "question": last_message,
        "data": [],
        "history": []
    }
    
    # 执行工作流
    response = await multi_tool_workflow.ainvoke(input_state)
    return {"messages": [AIMessage(content=response["answer"])]}

async def check_hallucinations(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Analyze the user's query and checks if the response is supported by the set of facts based on the document retrieved,
    providing a binary score result.

    This function uses a language model to analyze the user's query and gives a binary score result.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    model = create_agent_llm(temperature=0.7, tags=["hallucinations"])
    
    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=state.documents,
        generation=state.messages[-1]
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ] + state.messages

    logger.info("---CHECK HALLUCINATIONS---")
    
    response = cast(GradeHallucinations, await model.with_structured_output(GradeHallucinations).ainvoke(messages))
    
    return {"hallucination": response} 


# 定义持久化存储，也可以使用SQLiteSaver()、PostgresSaver()等
# LangGraph官方地址：https://langchain-ai.github.io/langgraph/how-tos/persistence/
checkpointer = MemorySaver()

# 定义状态图
builder = StateGraph(AgentState, input=InputState)
# 添加节点
builder.add_node(analyze_and_route_query)
builder.add_node(respond_to_general_query)
builder.add_node(get_additional_info)
builder.add_node("create_research_plan", create_research_plan)  # 这里是子图
builder.add_node(create_image_query)
builder.add_node(create_file_query)

# 添加边
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)


graph = builder.compile(checkpointer=checkpointer)

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))