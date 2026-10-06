"""
数据飞轮优化
记录回答失败/低置信度的问题，支持人工补充知识后重新学习
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# 数据飞轮目录
FLYWHEEL_DIR = Path(__file__).parent.parent / "graphrag" / "flywheel"
FLYWHEEL_DIR.mkdir(parents=True, exist_ok=True)


class FailureRecord:
    """失败记录"""
    
    def __init__(self, query: str, answer: str = None, confidence: float = 0.0, 
                 failure_type: str = "unknown", metadata: Dict = None):
        self.query = query
        self.answer = answer
        self.confidence = confidence
        self.failure_type = failure_type
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.id = f"fail_{int(time.time() * 1000)}"
        self.status = "pending"  # pending | resolved
        self.supplementary_knowledge = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "failure_type": self.failure_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "status": self.status,
            "supplementary_knowledge": self.supplementary_knowledge
        }
    
    def mark_resolved(self, knowledge: str):
        self.status = "resolved"
        self.supplementary_knowledge = knowledge
        self.resolved_at = datetime.now().isoformat()


class DataFlywheel:
    """
    数据飞轮服务
    目标: 让机器越来越聪明，少兜底"不知道"
    """
    
    RECORDS_FILE = FLYWHEEL_DIR / "failure_records.json"
    RESOLVED_FILE = FLYWHEEL_DIR / "resolved_knowledge.json"
    
    def __init__(self, auto_log_threshold: float = 0.6):
        """
        初始化数据飞轮
        
        Args:
            auto_log_threshold: 自动记录失败的置信度阈值
        """
        self.auto_log_threshold = auto_log_threshold
        self.records: List[FailureRecord] = []
        self._load_records()
    
    def _load_records(self):
        """加载历史记录"""
        if self.RECORDS_FILE.exists():
            try:
                with open(self.RECORDS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.records = [FailureRecord(**r) if isinstance(r, dict) and "query" in r else r for r in raw]
            except Exception:
                self.records = []
    
    def _save_records(self):
        """保存记录"""
        with open(self.RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() if hasattr(r, 'to_dict') else r for r in self.records], 
                      f, ensure_ascii=False, indent=2)
    
    def log_failure(self, query: str, answer: str = None, confidence: float = 0.0,
                    failure_type: str = "low_confidence", metadata: Dict = None):
        """
        记录一个失败的查询
        
        Args:
            query: 用户查询
            answer: 系统给出的答案（可能是低质量的）
            confidence: 置信度
            failure_type: 失败类型
            metadata: 额外元数据
        """
        record = FailureRecord(
            query=query,
            answer=answer,
            confidence=confidence,
            failure_type=failure_type,
            metadata=metadata or {}
        )
        self.records.append(record)
        self._save_records()
        print(f"[DataFlywheel] 记录失败: '{query[:30]}' (置信度: {confidence:.2f})")
        return record
    
    def should_log(self, confidence: float) -> bool:
        """判断是否需要记录"""
        return confidence < self.auto_log_threshold
    
    def get_pending_failures(self, limit: int = 100) -> List[FailureRecord]:
        """获取待处理的失败记录"""
        pending = [r for r in self.records if r.status == "pending"]
        return pending[:limit]
    
    def get_pending_count(self) -> int:
        """待处理数量"""
        return len([r for r in self.records if r.status == "pending"])
    
    def add_knowledge(self, failure_id: str, knowledge: str):
        """
        为失败记录补充知识
        人工补充后再重新学习
        """
        for record in self.records:
            if record.id == failure_id:
                record.mark_resolved(knowledge)
                self._save_records()
                
                # 同时追加到已解决知识库
                self._add_to_resolved_library(knowledge, record.query)
                
                print(f"[DataFlywheel] 补充知识: '{failure_id}' → '{knowledge[:50]}'")
                return True
        
        return False
    
    def _add_to_resolved_library(self, knowledge: str, original_query: str):
        """追加到已解决知识库"""
        resolved = {}
        if self.RESOLVED_FILE.exists():
            with open(self.RESOLVED_FILE, "r", encoding="utf-8") as f:
                resolved = json.load(f)
        
        # 将原始查询作为 key
        if original_query not in resolved:
            resolved[original_query] = []
        
        resolved[original_query].append({
            "knowledge": knowledge,
            "added_at": datetime.now().isoformat()
        })
        
        with open(self.RESOLVED_FILE, "w", encoding="utf-8") as f:
            json.dump(resolved, f, ensure_ascii=False, indent=2)
    
    def get_resolved_knowledge(self, query: str) -> Optional[List[str]]:
        """
        查询已补充的知识（用于 RAG 增强）
        这是数据飞轮的关键环节: 把人工补充的知识反馈回检索
        """
        if not self.RESOLVED_FILE.exists():
            return None
        
        with open(self.RESOLVED_FILE, "r", encoding="utf-8") as f:
            resolved = json.load(f)
        
        # 精确匹配或相似查询
        if query in resolved:
            return [item["knowledge"] for item in resolved[query]]
        
        # 简单包含匹配
        for q, items in resolved.items():
            if q in query or query in q:
                return [item["knowledge"] for item in items]
        
        return None