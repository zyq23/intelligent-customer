#!/usr/bin/env python3
"""
RAG 系统评估套件 v2.0
改进版：使用语义相似度计算相关性
"""
import os, sys, json, pickle, requests, numpy as np, faiss, re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from dataclasses import dataclass
import statistics

# 配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:21434")
EMBEDDING_MODEL = "nomic-embed-text"
USER_ID = "8e8dca41-982b-53a3-b298-7d6a2b0ed7ee"


@dataclass
class QuerySample:
    """评估样本"""
    question: str
    golden_contexts: List[str]  # 黄金标准上下文（人工标注的最相关文档块）
    categories: List[str]
    difficulty: str


@dataclass
class EvaluationResult:
    """单个查询的评估结果"""
    question: str
    retrieved_docs: List[Dict]
    retrieved_indices: List[int]
    
    # 检索质量指标
    hit_rate: float
    precision_at_k: List[float]
    recall_at_k: List[float]
    mrr: float
    ndcg_at_k: List[float]
    map_score: float  # Average Precision


class RAGEvaluator:
    """RAG 评估器"""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]
        self.rag = self._load_rag()
        self.metadata = self._load_metadata()
        
    def _load_rag(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from rag_query import SimpleVectorRAG
        rag = SimpleVectorRAG(USER_ID)
        rag.load()
        return rag
    
    def _load_metadata(self):
        rag_base_dir = Path(f"app/graphrag/data/output/{USER_ID}/vector_rag")
        with open(rag_base_dir / "metadata.pkl", "rb") as f:
            return pickle.load(f)
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=120
            )
            embeddings.append(resp.json()["embedding"])
        return np.array(embeddings)
    
    def _compute_semantic_similarity(self, query: str, context: str) -> float:
        """计算查询和上下文的语义相似度"""
        try:
            q_emb = self._embed_batch([query])[0]
            c_emb = self._embed_batch([context])[0]
            
            # 余弦相似度
            similarity = np.dot(q_emb, c_emb) / (
                np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
            )
            return similarity
        except Exception as e:
            print(f"  相似度计算错误：{e}")
            return 0.0
    
    def _compute_relevance_scores(self, query: str, golden_contexts: List[str]) -> List[float]:
        """
        计算所有文档块与黄金上下文的语义相关性
        
        方法：如果文档块与任意黄金上下文的相似度超过阈值，则认为相关
        """
        all_chunks = self.metadata["chunks"]
        relevance_scores = []
        
        # 先计算黄金上下文的平均嵌入
        if not golden_contexts:
            return [0.0] * len(all_chunks)
        
        gold_embeddings = self._embed_batch(golden_contexts)
        avg_gold_emb = np.mean(gold_embeddings, axis=0)
        
        # 计算每个文档块与黄金上下文的相似度
        threshold = 0.65  # 相似度阈值
        
        for chunk in all_chunks:
            chunk_emb = self._embed_batch([chunk])[0]
            similarity = np.dot(chunk_emb, avg_gold_emb) / (
                np.linalg.norm(chunk_emb) * np.linalg.norm(avg_gold_emb)
            )
            
            # 转换为 0-5 的相关性评分
            if similarity >= 0.85:
                score = 5
            elif similarity >= 0.75:
                score = 4
            elif similarity >= 0.65:
                score = 3
            elif similarity >= 0.55:
                score = 2
            elif similarity >= 0.45:
                score = 1
            else:
                score = 0
            
            relevance_scores.append(score)
        
        return relevance_scores
    
    def evaluate_query(self, sample: QuerySample, top_k: int = 10) -> EvaluationResult:
        """评估单个查询"""
        
        # 1. 获取 RAG 检索结果
        rag_results = self.rag.query(sample.question, top_k=top_k)
        retrieved_indices = []
        
        # 找到检索结果的索引
        for doc in rag_results:
            try:
                idx = self.metadata["chunks"].index(doc["content"])
                retrieved_indices.append(idx)
            except ValueError:
                pass
        
        # 2. 计算相关性评分
        relevance_scores = self._compute_relevance_scores(
            sample.question, 
            sample.golden_contexts
        )
        
        # 3. 计算评估指标
        k_values = [k for k in self.k_values if k <= top_k]
        
        # Hit Rate: Top-K 中是否有相关文档 (score >= 3)
        has_relevant = any(
            relevance_scores[idx] >= 3 
            for idx in retrieved_indices[:top_k]
        )
        hit_rate = 1.0 if has_relevant else 0.0
        
        # Precision@K, Recall@K, NDCG@K, MAP
        precisions = []
        recalls = []
        ndcgs = []
        aps = []  # Average Precision for each query
        
        total_relevant_docs = sum(1 for s in relevance_scores if s >= 3)
        
        for k in k_values:
            # 前 K 个检索结果
            top_k_indices = retrieved_indices[:k]
            
            # Precision@K
            relevant_in_top_k = sum(
                1 for idx in top_k_indices 
                if relevance_scores[idx] >= 3
            )
            precision = relevant_in_top_k / k if k > 0 else 0
            precisions.append(precision)
            
            # Recall@K
            recall = relevant_in_top_k / total_relevant_docs if total_relevant_docs > 0 else 0
            recalls.append(recall)
            
            # DCG@K 和 NDCG@K
            dcg = sum(
                relevance_scores[idx] / np.log2(i + 2)
                for i, idx in enumerate(top_k_indices)
            )
            
            # IDCG (理想排序)
            ideal_scores = sorted(relevance_scores, reverse=True)[:k]
            idcg = sum(
                score / np.log2(i + 2)
                for i, score in enumerate(ideal_scores)
            )
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        # MRR (Mean Reciprocal Rank)
        first_relevant_rank = None
        for i, idx in enumerate(retrieved_indices, 1):
            if relevance_scores[idx] >= 3:
                first_relevant_rank = i
                break
        
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        # MAP (Average Precision)
        ap = 0.0
        if total_relevant_docs > 0:
            num_relevant_so_far = 0
            for i, idx in enumerate(retrieved_indices[:min(top_k, len(retrieved_indices))], 1):
                if relevance_scores[idx] >= 3:
                    num_relevant_so_far += 1
                    precision_at_this_point = num_relevant_so_far / i
                    ap += precision_at_this_point
            ap = ap / min(total_relevant_docs, len(retrieved_indices[:top_k])) if retrieved_indices else 0
        
        result = EvaluationResult(
            question=sample.question,
            retrieved_docs=rag_results,
            retrieved_indices=retrieved_indices,
            hit_rate=hit_rate,
            precision_at_k=precisions,
            recall_at_k=recalls,
            mrr=mrr,
            ndcg_at_k=ndcgs,
            map_score=ap
        )
        
        return result
    
    def batch_evaluate(self, samples: List[QuerySample]) -> Dict:
        """批量评估"""
        results = []
        
        for i, sample in enumerate(samples):
            print(f"Evaluating {i+1}/{len(samples)}: {sample.question[:60]}...")
            result = self.evaluate_query(sample)
            results.append(result)
        
        # 聚合统计
        aggregated = {
            "total_samples": len(results),
            "avg_hit_rate": statistics.mean([r.hit_rate for r in results]),
            "avg_mrr": statistics.mean([r.mrr for r in results]),
            "avg_map": statistics.mean([r.map_score for r in results]),
        }
        
        # 按 K 值统计
        for idx, k in enumerate(self.k_values):
            if idx < len(results[0].precision_at_k):
                aggregated[f"avg_precision@{k}"] = statistics.mean([
                    r.precision_at_k[idx] for r in results if idx < len(r.precision_at_k)
                ])
                aggregated[f"avg_recall@{k}"] = statistics.mean([
                    r.recall_at_k[idx] for r in results if idx < len(r.recall_at_k)
                ])
                aggregated[f"avg_ndcg@{k}"] = statistics.mean([
                    r.ndcg_at_k[idx] for r in results if idx < len(r.ndcg_at_k)
                ])
        
        # 按难度分类
        aggregated["by_difficulty"] = {}
        for diff in ["easy", "medium", "hard"]:
            diff_hits = [
                r.hit_rate for s, r in zip(samples, results)
                if s.difficulty == diff
            ]
            if diff_hits:
                aggregated["by_difficulty"][diff] = statistics.mean(diff_hits)
        
        # 按类别分类
        aggregated["by_category"] = {}
        category_hits = {}
        for sample, result in zip(samples, results):
            for category in sample.categories:
                if category not in category_hits:
                    category_hits[category] = []
                category_hits[category].append(result.hit_rate)
        
        for cat, hits in category_hits.items():
            aggregated["by_category"][cat] = statistics.mean(hits)
        
        return aggregated, results


def create_evaluation_dataset():
    """
    创建评估数据集 - 使用实际文档内容作为黄金标准
    """
    rag_base_dir = Path(f"app/graphrag/data/output/{USER_ID}/vector_rag")
    with open(rag_base_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    chunks = metadata["chunks"]
    
    # 根据实际问题创建带有黄金上下文的评估集
    samples = [
        # 问题 1: 博士学制
        QuerySample(
            question="博士研究生基准学制是几年？",
            golden_contexts=[
                "本博士研究生基准学制为 4 年，具体以录取当年招生目录为准。在学制内未完成学业的，经批准可以适当延长在校学习年限"
            ],
            categories=["学制管理", "博士"],
            difficulty="easy"
        ),
        
        # 问题 2: 硕士实践
        QuerySample(
            question="硕士研究生专业实践的要求是什么？",
            golden_contexts=[
                "原则上应至少参与一个完整项目的研发实践。专业实践的考核通过与否需由导师、实践单位和学院三方共同确认",
                "本专业硕士研究生参加官方举办的创新创业实践并取得成果，或取得与专业相关的实践经历"
            ],
            categories=["硕士毕业要求", "实践"],
            difficulty="medium"
        ),
        
        # 问题 3: 论文答辩
        QuerySample(
            question="学位论文答辩的申请流程是怎样的？",
            golden_contexts=[
                "学位论文答辩申请",
                "答辩委员会组成",
                "答辩程序和要求",
                "学术论文评价办法"
            ],
            categories=["答辩要求", "论文"],
            difficulty="medium"
        ),
        
        # 问题 4: 科研成果
        QuerySample(
            question="研究生发表学术论文有什么要求？",
            golden_contexts=[
                "学术论文评价办法",
                "论文级别认定标准",
                "研究成果评价",
                "学术论文创新性要求"
            ],
            categories=["科研成果", "论文发表"],
            difficulty="medium"
        ),
        
        # 问题 5: 课程体系
        QuerySample(
            question="电子信息专业硕士研究生的培养方案包括哪些课程？",
            golden_contexts=[
                "公共必修课",
                "专业基础课",
                "专业核心课",
                "选修课程",
                "跨学科课程"
            ],
            categories=["课程设置", "培养方案"],
            difficulty="easy"
        ),
        
        # 问题 6: AI 方向
        QuerySample(
            question="人工智能领域的核心研究方向有哪些？",
            golden_contexts=[
                "移动计算智能",
                "智能感知技术",
                "上下文信息理解与融合",
                "移动智能控制系统",
                "虚拟现实与增强现实"
            ],
            categories=["研究方向", "人工智能"],
            difficulty="hard"
        ),
        
        # 问题 7: 毕业条件
        QuerySample(
            question="硕士研究生毕业需要满足哪些条件？",
            golden_contexts=[
                "修满培养方案规定的学分",
                "完成专业实践或实习",
                "通过硕士学位论文答辩",
                "达到学位授予的各项要求"
            ],
            categories=["毕业要求", "硕士"],
            difficulty="easy"
        ),
        
        # 问题 8: 博士科研成果
        QuerySample(
            question="博士研究生毕业对科研成果有什么具体要求？",
            golden_contexts=[
                "发表高水平学术论文",
                "学术论文评价标准",
                "研究成果创新性",
                "学术水平要求"
            ],
            categories=["毕业要求", "博士", "科研成果"],
            difficulty="hard"
        ),
        
        # 问题 9: 外出实践
        QuerySample(
            question="研究生外出学习和实践��管理规定是什么？",
            golden_contexts=[
                "研究生外出学习实践管理办法",
                "外出学习实践报告及考核评价",
                "实践审批流程",
                "实践要求"
            ],
            categories=["实践管理", "外出学习"],
            difficulty="medium"
        ),
        
        # 问题 10: 培养模式对比
        QuerySample(
            question="专硕和学硕的培养模式有什么不同？",
            golden_contexts=[
                "专业学位侧重工程实践",
                "学术学位侧重科学研究",
                "专业实践要求",
                "培养目标和方向差异"
            ],
            categories=["培养模式", "学位类型"],
            difficulty="hard"
        ),
        
        # 问题 11: 学分要求
        QuerySample(
            question="硕士研究生需要修满多少学分才能毕业？",
            golden_contexts=[
                "学分要求",
                "培养方案规定的总学分",
                "各类课程学分分配"
            ],
            categories=["学分要求", "毕业条件"],
            difficulty="easy"
        ),
        
        # 问题 12: 延期毕业
        QuerySample(
            question="学生可以申请延期毕业吗？有什么规定？",
            golden_contexts=[
                "在学制内未完成学业的可延长学习年限",
                "经批准可以适当延长",
                "延期毕业申请流程"
            ],
            categories=["学籍管理", "延期毕业"],
            difficulty="medium"
        )
    ]
    
    return samples


def generate_evaluation_report(aggregated: Dict, results: List[EvaluationResult], samples: List[QuerySample]):
    """生成详细的评估报告"""
    
    report = {
        "summary": aggregated,
        "detailed_results": [],
        "recommendations": []
    }
    
    # 详细结果
    for sample, result in zip(samples, results):
        report["detailed_results"].append({
            "question": sample.question,
            "difficulty": sample.difficulty,
            "categories": sample.categories,
            "hit_rate": result.hit_rate,
            "mrr": result.mrr,
            "map": result.map_score,
            "precision_at_k": dict(zip(evaluator.k_values, result.precision_at_k)),
            "recall_at_k": dict(zip(evaluator.k_values, result.recall_at_k)),
            "ndcg_at_k": dict(zip(evaluator.k_values, result.ndcg_at_k)),
            "top_retrieved": result.retrieved_docs[:3]
        })
    
    # 生成建议
    if aggregated["avg_hit_rate"] < 0.5:
        report["recommendations"].append("召回率低，考虑优化 embedding 模型或增加检索维度")
    if aggregated["avg_ndcg@5"] < 0.6:
        report["recommendations"].append("排序质量不佳，考虑引入重排序模型")
    if aggregated["by_difficulty"].get("hard", 0) < 0.3:
        report["recommendations"].append("复杂问题表现差，需要增强上下文理解和多跳推理能力")
    
    if not report["recommendations"]:
        report["recommendations"].append("系统表现良好，继续优化即可")
    
    return report


def main():
    print("="*70)
    print("RAG 系统标准化评估 v2.0")
    print("="*70)
    
    # 1. 创建评估集
    print("\n[1] 创建评估数据集...")
    samples = create_evaluation_dataset()
    print(f"✅ 共 {len(samples)} 个评估样本")
    
    # 2. 初始化评估器
    print("\n[2] 初始化评估器...")
    evaluator = RAGEvaluator(k_values=[1, 3, 5, 10])
    
    # 3. 批量评估
    print("\n[3] 开始评估...")
    aggregated, results = evaluator.batch_evaluate(samples)
    
    # 4. 生成报告
    report = generate_evaluation_report(aggregated, results, samples)
    
    # 5. 显示结果
    print("\n" + "="*70)
    print("评估结果汇总")
    print("="*70)
    
    print(f"\n【整体性能】")
    print(f"  总样本数：    {aggregated['total_samples']}")
    print(f"  Hit Rate:     {aggregated['avg_hit_rate']:.4f}")
    print(f"  MRR:          {aggregated['avg_mrr']:.4f}")
    print(f"  MAP:          {aggregated['avg_map']:.4f}")
    
    print(f"\n【Precision@K】")
    for k in evaluator.k_values:
        key = f"avg_precision@{k}"
        if key in aggregated:
            print(f"  P@{k}: {aggregated[key]:.4f}")
    
    print(f"\n【Recall@K】")
    for k in evaluator.k_values:
        key = f"avg_recall@{k}"
        if key in aggregated:
            print(f"  R@{k}: {aggregated[key]:.4f}")
    
    print(f"\n【NDCG@K】")
    for k in evaluator.k_values:
        key = f"avg_ndcg@{k}"
        if key in aggregated:
            print(f"  N@{k}: {aggregated[key]:.4f}")
    
    print(f"\n【按难度分类 - Hit Rate】")
    for diff, score in aggregated.get("by_difficulty", {}).items():
        print(f"  {diff}: {score:.4f}")
    
    print(f"\n【系统建议】")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    # 6. 保存结果
    output_dir = Path("app/graphrag/evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"evaluation_summary_{timestamp}.json"
    detailed_file = output_dir / f"evaluation_detailed_{timestamp}.json"
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估报告已保存到:")
    print(f"   - {summary_file}")
    print(f"   - {detailed_file}")
    
    return aggregated


if __name__ == "__main__":
    main()
