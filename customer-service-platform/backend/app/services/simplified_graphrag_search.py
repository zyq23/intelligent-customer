#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版 GraphRAG 查询器
使用预构建的文档向量进行语义搜索
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/data/zyq/intelligent-customer/customer-service-platform/backend')

from dotenv import load_dotenv
load_dotenv('/data/zyq/intelligent-customer/customer-service-platform/backend/.env')

import pandas as pd
import json

async def simple_search(query: str, top_k: int = 5):
    """
    简化版查询：从 documents.parquet 中进行关键词/语义匹配搜索
    """
    output_dir = Path("/data/zyq/intelligent-customer/customer-service-platform/backend/app/graphrag/data/output/8e8dca41-982b-53a3-b298-7d6a2b0ed7ee")
    documents_path = output_dir / "documents.parquet"
    
    if not documents_path.exists():
        return None, "索引文件不存在"
    
    # 加载文档
    df = pd.read_parquet(documents_path)
    print(f"加载了 {len(df)} 条文档记录")
    
    # 简单关键词匹配（也可以使用向量相似度）
    query_lower = query.lower()
    keywords = query_lower.split()
    
    # 查找包含关键词的文档
    matches = []
    for _, row in df.iterrows():
        text = str(row.get('text', '')) + str(row.get('metadata', ''))
        text_lower = text.lower()
        
        # 计算匹配度
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            matches.append({
                'score': score,
                'text': text[:1000],  # 限制长度
                'id': row.get('id', '')
            })
    
    # 按匹配度排序
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    if not matches:
        # 如果没有匹配，尝试向量相似度搜索
        try:
            import lancedb
            from sentence_transformers import SentenceTransformer
            
            # 使用轻量模型
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L3-v2')
            
            db = lancedb.connect(str(output_dir / "text_lancedb"))
            table = db.open_table('vector_search_table')
            
            # 生成查询向量
            query_vector = model.encode([query])
            
            # 检索相似文档
            results = table.search(query_vector[0]).limit(top_k).to_pandas()
            
            for _, row in results.iterrows():
                matches.append({
                    'score': 1.0,
                    'text': str(row.get('text', row.get('document', '')))[:1000],
                    'id': row.get('id', '')
                })
        except Exception as e:
            print(f"向量搜索失败: {e}")
            return None, f"搜索出错: {e}"
    
    return matches[:top_k], None


async def search_documents(query: str):
    """查询文档"""
    print(f"\n{'='*60}")
    print(f"🔍 查询: {query}")
    print('='*60)
    
    results, error = await simple_search(query)
    
    if error:
        print(f"❌ 错误: {error}")
        return
    
    if not results:
        print("⚠️ 未找到相关文档")
        return
    
    print(f"\n✅ 找到 {len(results)} 条相关结果：\n")
    
    for i, match in enumerate(results, 1):
        score = match['score'] if isinstance(match['score'], str) or score > 1 else match['score']
        print(f"--- 结果 {i} (匹配度: {score:.2f}/5) ---")
        print(match['text'][:500])
        if len(match['text']) > 500:
            print(f"...（{len(match['text'])-500} 字符省略）")
        print()


def main():
    """测试查询"""
    questions = [
        "人工智能专业硕士的培养目标是什么？",
        "硕士毕业需要多少学分？",
        "科研成果要求有哪些？",
        "申请学位答辩需要什么材料？",
    ]
    
    async def run_tests():
        for q in questions:
            await search_documents(q)
            import time
            time.sleep(1)
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()