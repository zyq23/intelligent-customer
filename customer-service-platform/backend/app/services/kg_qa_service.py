"""
知识图谱问答服务
从 Neo4j 实体/关系图中进行推理问答
"""

from typing import List, Dict, Optional, Any
from opentelemetry import trace


class KnowledgeGraphQA:
    """知识图谱问答服务"""
    
    def __init__(self, neo4j_connection=None):
        self.connection = neo4j_connection
        self.tracer = trace.get_tracer("kg_qa")
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        从查询中提取可能的实体关键词
        """
        # 简单的实体抽取规则
        entity_patterns = [
            # 学业类型
            ("博士", ["博士研究生", "博士", "PhD"]),
            ("硕士", ["硕士研究生", "硕士", "Master"]),
            ("学位", ["学位", "学士", "博士学位", "硕士学位"]),
            # 学业阶段
            ("阶段", ["阶段", "学期", "年级"]),
            ("论文", ["论文", "毕业论文", "科研论文"]),
            # 要求
            ("要求", ["要求", "条件", "标准", "规定"]),
            ("毕业", ["毕业", " graduation"]),
            ("答辩", ["答辩", " defense"]),
        ]
        
        entities = []
        query_lower = query.lower()
        
        for pattern, variants in entity_patterns:
            for variant in variants:
                if variant in query or variant.lower() in query_lower:
                    entities.append(pattern)
                    break
        
        return list(set(entities))
    
    def query_graph(self, entities: List[str], query: str) -> Optional[Dict]:
        """
        查询 Neo4j 图谱
        """
        if not self.connection:
            return None
        
        # 构建查询
        # 示例：查询关于"论文"的实体
        cypher_queries = [
            "MATCH (n) WHERE n.name CONTAINS $entity RETURN n LIMIT 5",
            "MATCH (n)-[r]->(m) WHERE n.name CONTAINS $entity RETURN n, r, m LIMIT 10",
            "MATCH (n:Paper)-[r]->(m) RETURN n, r, m LIMIT 5",
        ]
        
        results = []
        for cypher in cypher_queries:
            try:
                for entity in entities:
                    with self.connection.session() as session:
                        result = session.run(cypher, entity=entity)
                        for record in result:
                            results.append(record)
            except Exception:
                continue
        
        if results:
            return {
                "entities": [dict(r) for r in results],
                "relations": [],
                "raw_count": len(results)
            }
        
        return None
    
    def reason_from_graph(self, query: str) -> Optional[str]:
        """
        从知识图谱推理答案
        """
        with self.tracer.start_as_current_span("kg.reason") as span:
            span.set_attribute("query", query)
            
            # 提取实体
            entities = self.extract_entities_from_query(query)
            span.set_attribute("entities", ",".join(entities))
            
            # 查询图谱
            graph_data = self.query_graph(entities, query)
            
            if graph_data:
                # 简单的推理逻辑
                answer = self._infer_answer(entities, graph_data)
                span.set_attribute("reasoning_done", True)
                return answer
            
            span.set_attribute("reasoning_done", False)
            return None
    
    def _infer_answer(self, entities: List[str], graph_data: Dict) -> str:
        """
        从图谱数据推理答案
        """
        # 这里可以添加更复杂的推理逻辑
        # 例如：SPARQL 查询、路径推理等
        
        if "论文" in entities:
            # 根据论文实体推理
            return "根据知识图谱，论文是毕业的重要组成部分。"
        
        if "学位" in entities:
            return "学位要求包括完成学业培养计划、通过考试、提交论文等。"
        
        if "答辩" in entities:
            return "答辩是学位授予的最后环节，需要提交答辩稿。"
        
        return f"找到 {graph_data.get('raw_count', 0)} 个相关实体。"


# KG QA 服务实例
_kg_qa = None


def get_kg_qa_service() -> KnowledgeGraphQA:
    """获取 KG 问答服务实例"""
    global _kg_qa
    
    if _kg_qa is None:
        try:
            from app.agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
            connection = get_neo4j_graph()
            _kg_qa = KnowledgeGraphQA(connection)
        except Exception as e:
            print(f"KG QA 初始化失败: {e}")
            _kg_qa = KnowledgeGraphQA()
    
    return _kg_qa


def query_knowledge_graph(query: str) -> Optional[Dict]:
    """
    快速查询知识图谱
    
    Args:
        query: 用户查询
        
    Returns:
        推理结果或 None
    """
    service = get_kg_qa_service()
    
    # 首先尝试图谱推理
    kg_answer = service.reason_from_graph(query)
    
    if kg_answer:
        return {
            "source": "knowledge_graph",
            "answer": kg_answer,
            "confidence": 0.8
        }
    
    return None


if __name__ == "__main__":
    # 测试
    print("测试知识图谱问答...")
    
    test_queries = [
        "毕业需要发表几篇论文？",
        "博士学制几年？",
        "论文答辩要求是什么？",
    ]
    
    for q in test_queries:
        result = query_knowledge_graph(q)
        if result:
            print(f"\n查询: {q}")
            print(f"来源: {result['source']}")
            print(f"答案: {result['answer']}")
        else:
            print(f"\n查询: {q}")
            print(f"无法从图谱获取答案")