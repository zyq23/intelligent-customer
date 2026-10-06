"""文档管理 API - 知识库文件上传、列表和删除"""
import os
import shutil
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.logger import get_logger
from app.models.user import User
from app.services.user_service import UserService

logger = get_logger(service="api.documents")
router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _format_size(size: int) -> str:
    """格式化文件大小"""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    else:
        return f"{size / (1024 * 1024):.2f} MB"


def _get_file_type(suffix: str) -> str:
    """根据文件扩展名判断类型"""
    type_map = {
        ".pdf": "PDF",
        ".txt": "文本",
        ".md": "Markdown",
        ".doc": "Word",
        ".docx": "Word",
        ".csv": "CSV",
        ".xls": "Excel",
        ".xlsx": "Excel",
        ".png": "图片",
        ".jpg": "图片",
        ".jpeg": "图片",
        ".gif": "图片",
        ".zip": "压缩包",
        ".rar": "压缩包",
    }
    return type_map.get(suffix.lower(), "其他")


@router.get("/user/{user_id}")
async def list_documents(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户的所有文档列表"""
    try:
        # 验证用户权限
        if current_user.id != user_id and not getattr(current_user, 'is_admin', False):
            raise HTTPException(status_code=403, detail="无权访问其他用户的文档")
        
        user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
        user_dir = UPLOAD_DIR / user_uuid
        
        documents = []
        
        if user_dir.exists():
            for timestamp_dir in sorted(user_dir.iterdir(), reverse=True):
                if timestamp_dir.is_dir():
                    for file_path in timestamp_dir.glob("*"):
                        if file_path.is_file():
                            parts = file_path.stem.rsplit("_", 1)
                            original_name = parts[0] if len(parts) > 1 else file_path.stem
                            
                            stat = file_path.stat()
                            
                            doc = {
                                "filename": file_path.name,
                                "original_name": original_name,
                                "size": stat.st_size,
                                "formatted_size": _format_size(stat.st_size),
                                "path": str(file_path).replace("\\", "/"),
                                "user_id": user_id,
                                "upload_time": timestamp_dir.name,
                                "directory": str(timestamp_dir).replace("\\", "/"),
                                "file_type": _get_file_type(file_path.suffix)
                            }
                            documents.append(doc)
        
        logger.info(f"Listed {len(documents)} documents for user {user_id}")
        return {"documents": documents, "total": len(documents)}
    
    except PermissionError:
        raise HTTPException(status_code=403, detail="无权访问此文档")
    except Exception as e:
        logger.error(f"Error listing documents for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败：{str(e)}")


@router.delete("/{file_path:path}")
async def delete_document(
    file_path: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除文档及其文件"""
    try:
        decoded_path = urllib.parse.unquote(file_path)
        normalized_path = decoded_path.replace("/", os.sep).replace("\\", os.sep)
        full_path = UPLOAD_DIR / normalized_path
        
        try:
            full_path_resolved = full_path.resolve(strict=False)
            uploads_resolved = UPLOAD_DIR.resolve()
            
            if not str(full_path_resolved).startswith(str(uploads_resolved)):
                raise HTTPException(status_code=403, detail="无效的文件路径")
        except Exception:
            raise HTTPException(status_code=403, detail="无效的文件路径")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        os.remove(full_path)
        logger.info(f"Document deleted: {full_path}")
        
        parent_dir = full_path.parent
        if parent_dir.exists() and parent_dir != UPLOAD_DIR:
            try:
                if not any(parent_dir.iterdir()):
                    shutil.rmtree(parent_dir)
                    logger.info(f"Empty directory removed: {parent_dir}")
            except OSError:
                pass
        
        return {"success": True, "message": "文档已成功删除"}
    
    except PermissionError:
        raise HTTPException(status_code=403, detail="无权删除此文档")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="文件不存在")
    except Exception as e:
        logger.error(f"Failed to delete document {file_path}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除文档失败：{str(e)}")
