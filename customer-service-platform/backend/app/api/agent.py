"""LangGraph Agent 接口: 智能客服主对话流"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from app.agent.lg_builder import graph
from app.agent.lg_states import InputState
from app.agent.utils import new_uuid
from app.core.logger import get_logger
from app.schemas.chat import LangGraphResumeRequest

logger = get_logger(service="api.agent")

router = APIRouter(prefix="/langgraph", tags=["agent"])


def _stream_messages(astream_agen, thread_config: dict, thread_id: str):
    """将 LangGraph 消息流转换为 SSE 流"""
    async def process_stream():
        async for c, metadata in astream_agen:
            # 只处理最终展示给用户的内容, 跳过中间工具调用和内部状态
            if c.content and "research_plan" not in metadata.get("tags", []) and not c.additional_kwargs.get("tool_calls"):
                content_json = json.dumps(c.content, ensure_ascii=False)
                yield f"data: {content_json}\n\n"
            elif c.additional_kwargs.get("tool_calls"):
                tool_data = c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
                logger.debug(f"Tool call: {tool_data}")

        # 处理中断情况
        state = graph.get_state(thread_config)
        if len(state) > 0 and len(state[-1]) > 0:
            if len(state[-1][0].interrupts) > 0:
                interrupt_json = json.dumps({"interruption": True, "conversation_id": thread_id})
                yield f"data: {interrupt_json}\n\n"

    return process_stream()


@router.post("/query")
async def langgraph_query(
    query: str = Form(...),
    user_id: int = Form(...),
    conversation_id: str = Form(None),
    image: UploadFile = File(None),
    file: UploadFile = File(None),
):
    """使用LangGraph处理用户查询, 支持图片上传(SSE流式)"""
    try:
        logger.info(f"Processing LangGraph query for user {user_id} and conversation {conversation_id}")

        # 处理图片上传
        image_path = None
        if image:
            image_dir = Path("uploads/images")
            image_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name, ext = os.path.splitext(image.filename)
            new_filename = f"{original_name}_{timestamp}{ext}"
            image_path = image_dir / new_filename

            content = await image.read()
            with open(image_path, "wb") as f:
                f.write(content)
            logger.info(f"Saved image {new_filename} for user {user_id}")

        # 处理文件上传 (file-query 分支: 让客服能分析用户上传的文件)
        file_path = None
        if file:
            user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
            first_level = Path("uploads") / user_uuid
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            second_level = first_level / timestamp
            second_level.mkdir(parents=True, exist_ok=True)

            original_name, ext = os.path.splitext(file.filename)
            new_filename = f"{original_name}_{timestamp}{ext}"
            file_path = second_level / new_filename

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(
                f"Saved file {new_filename} for user {user_id} (will route to file-query)"
            )

        # 使用conversation_id作为thread_id, 如果没有提供则创建新的
        thread_id = conversation_id if conversation_id else new_uuid()
        thread_config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "image_path": str(image_path) if image_path else None,
                "file_path": str(file_path) if file_path else None,
            }
        }

        # 获取当前线程状态
        state_history = None
        try:
            if thread_id:
                state_history = graph.get_state(thread_config)
                if state_history:
                    logger.info(f"Found existing conversation state for thread_id: {thread_id}")
        except Exception as e:
            logger.warning(f"Error retrieving state: {e}. Starting with fresh state.")

        # 准备输入状态 - 如果是现有会话, 直接传入查询文本
        if state_history and len(state_history) > 0 and len(state_history[-1]) > 0:
            logger.info("Using existing conversation state")
            agen = graph.astream(
                Command(resume=query),
                stream_mode="messages",
                config=thread_config,
            )
        else:
            logger.info("Creating new conversation state")
            input_state = InputState(messages=query)
            agen = graph.astream(
                input=input_state,
                stream_mode="messages",
                config=thread_config,
            )

        response = StreamingResponse(
            _stream_messages(agen, thread_config, thread_id),
            media_type="text/event-stream",
        )
        response.headers["X-Conversation-ID"] = thread_id
        return response

    except Exception as e:
        logger.error(f"LangGraph query error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume")
async def langgraph_resume(request: LangGraphResumeRequest):
    """继续执行LangGraph流程(处理中断)"""
    try:
        logger.info(f"Resuming LangGraph query for user {request.user_id} with conversation {request.conversation_id}")

        # 使用会话ID作为线程ID
        thread_config = {"configurable": {"thread_id": request.conversation_id}}

        async def process_resume():
            async for c, metadata in graph.astream(
                Command(resume=request.query), stream_mode="messages", config=thread_config
            ):
                # 只处理最终展示给用户的内容
                if c.content and not c.additional_kwargs.get("tool_calls"):
                    content_json = json.dumps(c.content, ensure_ascii=False)
                    yield f"data: {content_json}\n\n"
                elif c.additional_kwargs.get("tool_calls"):
                    tool_data = c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
                    logger.debug(f"Tool call: {tool_data}")

        return StreamingResponse(process_resume(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"LangGraph resume error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
