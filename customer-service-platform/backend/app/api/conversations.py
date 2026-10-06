"""会话管理接口: 创建 / 查询 / 删除 / 重命名"""
from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger
from app.schemas.chat import CreateConversationRequest, UpdateConversationNameRequest
from app.services.conversation_service import ConversationService

logger = get_logger(service="api.conversations")

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("")
async def create_conversation(request: CreateConversationRequest):
    """创建新会话"""
    try:
        conversation_id = await ConversationService.create_conversation(request.user_id)
        return {"conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_conversations(user_id: int):
    """获取用户的所有会话"""
    try:
        conversations = await ConversationService.get_user_conversations(user_id)
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: int, user_id: int):
    """获取会话的所有消息"""
    try:
        messages = await ConversationService.get_conversation_messages(conversation_id, user_id)
        return messages
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: int):
    """删除会话及其所有消息"""
    try:
        conversation_service = ConversationService()
        await conversation_service.delete_conversation(conversation_id)
        return {"message": "会话已删除"}
    except Exception as e:
        logger.error(f"删除会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{conversation_id}/name")
async def update_conversation_name(
    conversation_id: int,
    request: UpdateConversationNameRequest,
):
    """修改会话名称"""
    try:
        conversation_service = ConversationService()
        await conversation_service.update_conversation_name(conversation_id, request.name)
        return {"message": "会话名称已更新"}
    except Exception as e:
        logger.error(f"更新会话名称失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
