import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from app.core.config import settings

# 设置 GraphRAG 环境变量
if hasattr(settings, 'GRAPHRAG_API_KEY') and settings.GRAPHRAG_API_KEY:
    os.environ['GRAPHRAG_API_KEY'] = settings.GRAPHRAG_API_KEY
if hasattr(settings, 'GRAPHRAG_API_BASE') and settings.GRAPHRAG_API_BASE:
    os.environ['GRAPHRAG_API_BASE'] = settings.GRAPHRAG_API_BASE
if hasattr(settings, 'GRAPHRAG_MODEL_NAME') and settings.GRAPHRAG_MODEL_NAME:
    os.environ['GRAPHRAG_MODEL_NAME'] = settings.GRAPHRAG_MODEL_NAME

import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from app.core.logger import get_logger

logger = get_logger(service="search")


class SearchService:
    def __init__(self):
        self.project_dir = settings.GRAPHRAG_PROJECT_DIR
        self.data_dir_name = settings.GRAPHRAG_DATA_DIR
        self.data_dir = os.path.join(self.project_dir, self.data_dir_name)
        
        # 用户特定的输出目录映射
        self._output_dirs: Dict[int, str] = {}  # user_id -> output_dir

    def _get_user_output_dir(self, user_id: int) -> Optional[str]:
        """获取用户的输出目录"""
        return self._output_dirs.get(user_id)
    
    def _register_user_output_dir(self, user_id: int, output_dir: str):
        """注册用户的输出目录"""
        self._output_dirs[user_id] = output_dir
        logger.info(f"已为用户 {user_id} 注册输出目录：{output_dir}")

    def _load_vector_stores(self, output_dir: str):
        """加载向量存储"""
        try:
            # 加载文本嵌入
            text_embed_collection = LanceDBVectorStore(
                collection_name="create_final_chunks"
            )
            db_path = os.path.join(output_dir, "lancedb", "create_final_chunks")
            if os.path.exists(db_path):
                text_embed_collection.connect(db_uri=db_path)
                logger.info("成功连接到文本向量存储")
            else:
                logger.warning(f"文本向量存储路径不存在：{db_path}")
                text_embed_collection = None
            
            return {
                'text_units': text_embed_collection,
                'entities': text_embed_collection,
                'relationships': text_embed_collection,
                'communities': text_embed_collection,
                'community_reports': text_embed_collection,
                'covariates': None,  # 如果不需要协变量
            }
        except Exception as e:
            logger.error(f"加载向量存储失败：{str(e)}", exc_info=True)
            raise

    async def search(self, query: str, user_id: int = 0, search_type: str = "local") -> Dict[str, Any]:
        """
        执行搜索查询
        
        Args:
            query: 搜索查询
            user_id: 用户 ID
            search_type: 搜索类型 (local/global/drift)
            
        Returns:
            搜索结果字典
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"执行搜索查询：{query[:50]}..., 类型：{search_type}, 用户 ID: {user_id}")
                
                output_dir = self._get_user_output_dir(user_id)
                if not output_dir:
                    raise ValueError(f"未找到用户 {user_id} 的输出目录")
                
                # 设置配置
                config_path = os.path.join(self.data_dir, "settings.yaml")
                config_overrides = {
                    'output.base_dir': output_dir.replace('\\', '/'),
                }
                
                graphrag_config = load_config(
                    Path(self.data_dir),
                    Path(config_path),
                    config_overrides
                )
                
                # 加载向量存储
                vector_stores = self._load_vector_stores(output_dir)
                
                # 添加超时控制
                try:
                    if search_type.lower() == "local":
                        response = await asyncio.wait_for(
                            api.local_search(
                                config=graphrag_config,
                                entities=vector_stores['entities'],
                                communities=vector_stores['communities'],
                                community_reports=vector_stores['community_reports'],
                                text_units=vector_stores['text_units'],
                                relationships=vector_stores['relationships'],
                                covariates=vector_stores['covariates'],
                                community_level=getattr(settings, 'GRAPHRAG_COMMUNITY_LEVEL', 3),
                                response_type=getattr(settings, 'GRAPHRAG_RESPONSE_TYPE', 'multiple choices'),
                                query=query,
                            ),
                            timeout=120  # 最大等待时间 2 分钟
                        )
                    elif search_type.lower() == "global":
                        response = await asyncio.wait_for(
                            api.global_search(
                                config=graphrag_config,
                                entities=vector_stores['entities'],
                                communities=vector_stores['communities'],
                                community_reports=vector_stores['community_reports'],
                                text_units=vector_stores['text_units'],
                                relationships=vector_stores['relationships'],
                                covariates=vector_stores['covariates'],
                                community_level=getattr(settings, 'GRAPHRAG_COMMUNITY_LEVEL', 3),
                                response_type=getattr(settings, 'GRAPHRAG_RESPONSE_TYPE', 'multiple choices'),
                                query=query,
                            ),
                            timeout=120
                        )
                    elif search_type.lower() == "drift":
                        response = await asyncio.wait_for(
                            api.drift_search(
                                config=graphrag_config,
                                entities=vector_stores['entities'],
                                communities=vector_stores['communities'],
                                community_reports=vector_stores['community_reports'],
                                text_units=vector_stores['text_units'],
                                relationships=vector_stores['relationships'],
                                covariates=vector_stores['covariates'],
                                community_level=getattr(settings, 'GRAPHRAG_COMMUNITY_LEVEL', 3),
                                response_type=getattr(settings, 'GRAPHRAG_RESPONSE_TYPE', 'multiple choices'),
                                query=query,
                            ),
                            timeout=120
                        )
                    else:
                        raise ValueError(f"不支持的搜索类型：{search_type}")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"查询超时 (尝试 {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        raise TimeoutError("查询超时，请稍后重试")
                
                # 格式化响应
                result = {
                    'response': str(response.response) if hasattr(response, 'response') else str(response),
                    'context': {
                        'sources': [],
                        'entities': [],
                        'relationships': [],
                    },
                    'query': query,
                    'search_type': search_type,
                    'user_id': user_id
                }
                
                # 尝试提取上下文数据
                if hasattr(response, 'context_data') and response.context_data:
                    result['context'] = {
                        'sources': response.context_data.get('sources', []),
                        'entities': response.context_data.get('entities', []),
                        'relationships': response.context_data.get('relationships', []),
                    }
                
                logger.info(f"搜索完成 (尝试 {attempt + 1}/{max_retries})")
                return result
                
            except Exception as e:
                logger.warning(f"搜索失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"搜索最终失败 (经过 {max_retries} 次尝试): {str(e)}", exc_info=True)
                    return {
                        'status': 'error',
                        'error': f'搜索失败：{str(e)}',
                        'query': query,
                        'search_type': search_type,
                        'user_id': user_id
                    }

    def register_output_dir(self, user_id: int, output_dir: str):
        """注册用户的输出目录"""
        self._register_user_output_dir(user_id, output_dir)

    def clear_user_cache(self, user_id: int):
        """清除特定用户的缓存"""
        if user_id in self._output_dirs:
            del self._output_dirs[user_id]
            logger.info(f"已清除用户 {user_id} 的缓存")

    def clear_all_cache(self):
        """清除所有用户的缓存"""
        self._output_dirs.clear()
        logger.info("已清除所有用户缓存")
