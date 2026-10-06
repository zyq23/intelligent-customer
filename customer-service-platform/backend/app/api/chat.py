"""聊天相关接口: 普通对话 / 深度思考 / 联网检索"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.logger import get_logger, log_structured
from app.schemas.chat import ChatMessage, ReasonRequest
from app.services.llm_factory import LLMFactory
from app.services.conversation_service import ConversationService

logger = get_logger(service="api.chat")

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat_endpoint(request: ChatMessage):
    """普通对话接口(流式)"""
    try:
        logger.info(
            f"Processing chat request for user {request.user_id} in conversation {request.conversation_id}"
        )
        chat_service = LLMFactory.create_chat_service()

        return StreamingResponse(
            chat_service.generate_stream(
                messages=request.messages,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                on_complete=ConversationService.save_message,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reason")
async def reason_endpoint(request: ReasonRequest):
    """深度思考/推理接口(流式)"""
    try:
        logger.info(f"Processing reasoning request for user {request.user_id}")
        reasoner = LLMFactory.create_reasoner_service()

        log_structured(
            "reason_request",
            {
                "user_id": request.user_id,
                "message_count": len(request.messages),
                "last_message": request.messages[-1]["content"][:100] + "...",
            },
        )

        return StreamingResponse(
            reasoner.generate_stream(request.messages),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error(f"Reasoning error for user {request.user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_endpoint(request: ChatMessage):
    """带联网搜索的聊天接口(流式)"""
    try:
        logger.info(
            f"Processing search request for user {request.user_id} in conversation {request.conversation_id}"
        )
        logger.info(f"Request: {request}")
        search_service = LLMFactory.create_search_service()
        return StreamingResponse(
            search_service.generate_stream(
                query=request.messages[0]["content"],
                user_id=request.user_id,
                conversation_id=request.conversation_id,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
