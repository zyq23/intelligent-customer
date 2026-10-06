"""文件/图片上传接口"""
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.logger import get_logger
from app.services.indexing_service import IndexingService

logger = get_logger(service="api.uploads")

router = APIRouter(tags=["uploads"])

# 使用绝对路径，确保在不同工作目录下都能正确定位
BASE_DIR = Path(__file__).parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: int = Form(...)):
    """上传文件并准备 RAG 处理"""
    try:
        logger.info(f"Uploading file for user {user_id}: {file.filename}")

        # 1. 创建基于UUID的一级目录
        user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
        first_level_dir = UPLOAD_DIR / user_uuid

        # 2. 创建基于时间戳的二级目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        second_level_dir = first_level_dir / timestamp
        second_level_dir.mkdir(parents=True, exist_ok=True)

        # 3. 生成带时间戳的文件名
        original_name, ext = os.path.splitext(file.filename)
        new_filename = f"{original_name}_{timestamp}{ext}"
        file_path = second_level_dir / new_filename

        # 保存文件
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        file_info = {
            "filename": new_filename,
            "original_name": file.filename,
            "size": len(content),
            "type": file.content_type,
            "path": str(file_path).replace("\\", "/"),
            "user_id": user_id,
            "user_uuid": user_uuid,
            "upload_time": timestamp,
            "directory": str(second_level_dir),
        }

        # 4. 处理文件索引
        indexing_service = IndexingService()
        index_result = await indexing_service.process_file(file_info)

        return {**file_info, "index_result": index_result}
    except Exception as e:
        logger.error(f"Upload failed for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/image")
async def upload_image(
    image: UploadFile = File(...),
    user_id: int = Form(...),
    conversation_id: str = Form(None),
):
    """上传图片并返回图片存储路径"""
    try:
        # 创建图片存储目录
        image_dir = Path("uploads/images")
        if conversation_id:
            image_dir = image_dir / conversation_id
        image_dir.mkdir(parents=True, exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name, ext = os.path.splitext(image.filename)
        new_filename = f"{original_name}_{timestamp}{ext}"
        image_path = image_dir / new_filename

        # 保存图片
        content = await image.read()
        with open(image_path, "wb") as f:
            f.write(content)

        image_info = {
            "filename": new_filename,
            "original_name": image.filename,
            "size": len(content),
            "type": image.content_type,
            "path": str(image_path).replace("\\", "/"),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "upload_time": timestamp,
        }
        logger.info(f"Image uploaded: {image_info}")
        return image_info
    except Exception as e:
        logger.error(f"Image upload failed for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
