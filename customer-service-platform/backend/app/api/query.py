"""RAG 查询接口"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.logger import get_logger

logger = get_logger(service="api.query")
router = APIRouter(tags=["query"])


@router.post("/query")
async def query_endpoint(request_body: dict):
    """
    RAG 查询接口
    支持 BGE-M3 向量检索 + DeepSeek 生成式回复 + 数据飞轮
    
    用例:
    ```bash
    curl -X POST http://localhost:9002/api/query \
      -H "Content-Type: application/json" \
      -d '{"query": "博士学制几年？", "top_k": 5}'
    ```
    """
    query_text = request_body.get("query", "")
    top_k = request_body.get("top_k", 5)
    user_id = request_body.get("user_id", None)
    
    try:
        logger.info(f"RAG query: {query_text[:50]}...")
        
        from app.services.rag_query_service import query_rag
        results = query_rag(query_text, top_k=top_k, user_id=user_id)
        
        logger.info(f"Query found {results['total_results']} results, confidence: {results.get('confidence', 0):.2f}")
        
        return {
            "query": results["query"],
            "results": results["results"],
            "answer": results.get("answer", ""),
            "total_results": results["total_results"],
            "confidence": results.get("confidence", 0),
            "flywheel_hit": results.get("flywheel_hit", False)
        }
        
    except Exception as e:
        logger.error(f"RAG query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/test")
async def debug_test_rag():
    """
    测试 RAG 系统健康状况
    """
    try:
        from app.services.rag_query_service import query_rag
        
        test_query = "博士学制几年？"
        results = query_rag(test_query, top_k=3)
        
        return {
            "status": "ok",
            "test_query": test_query,
            "results_count": results["total_results"],
            "first_result_score": results["results"][0]["score"] if results["results"] else 0,
            "confidence": results.get("confidence", 0),
            "flywheel_hit": results.get("flywheel_hit", False),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/query/flywheel/pending")
async def flywheel_pending():
    """
    查看数据飞轮待处理的失败记录（管理员）
    """
    from app.services.data_flywheel import DataFlywheel
    
    flywheel = DataFlywheel()
    records = flywheel.get_pending_failures()
    
    return {
        "pending_count": len(records),
        "records": [
            {
                "id": r.id,
                "query": r.query,
                "confidence": r.confidence,
                "failure_type": r.failure_type,
                "timestamp": r.timestamp,
                "answer_preview": (r.answer or "")[:100]
            }
            for r in records
        ]
    }


@router.post("/query/flywheel/supplement")
async def flywheel_supplement(request_body: dict):
    """
    为失败记录补充知识（数据飞轮闭环）
    
    用例:
    ```bash
    curl -X POST http://localhost:9002/api/query/flywheel/supplement \
      -H "Content-Type: application/json" \
      -d '{"failure_id": "fail_xxx", "knowledge": "补充的知识内容"}'
    ```
    """
    from app.services.data_flywheel import DataFlywheel
    
    failure_id = request_body.get("failure_id")
    knowledge = request_body.get("knowledge", "").strip()
    
    if not failure_id or not knowledge:
        raise HTTPException(status_code=400, detail="需要 failure_id 和 knowledge")
    
    flywheel = DataFlywheel()
    ok = flywheel.add_knowledge(failure_id, knowledge)
    
    if not ok:
        raise HTTPException(status_code=404, detail="未找到该失败记录")
    
    return {"status": "ok", "failure_id": failure_id, "supplemented": True}