#!/usr/bin/env python3
"""快速 RAG 评估脚本"""
import os, sys, json, pickle, requests, numpy as np
from pathlib import Path
from datetime import datetime
import statistics

OLLAMA_URL = "http://127.0.0.1:21434"
USER_ID = "8e8dca41-982b-53a3-b298-7d6a2b0ed7ee"

# 测试问题 - 每个问题都有黄金标准答案
TEST_QUERIES = [
    ("博士学制几年？", ["基准学制为 4 年"], "easy"),
    ("硕士专业实践要求？", ["参与完整项目的研发实践", "导师、实践单位和学院三方确认"], "medium"),
    ("毕业需要满足什么条件？", ["修满培养方案规定的学分", "通过论文答辩"], "easy"),
    ("论文答辩怎么申请？", ["答辩申请", "答辩委员会组成"], "medium"),
    ("科研成果有什么要求？", ["发表论文", "学术论文评价办法"], "hard"),
]

def embed(text):
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings",
                        json={"model": "nomic-embed-text", "prompt": text},
                        timeout=120)
    return np.array(resp.json()["embedding"])

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("="*60)
    print("快速 RAG 评估")
    print("="*60)
    
    # 加载元数据
    rag_dir = Path(f"app/graphrag/data/output/{USER_ID}/vector_rag")
    with open(rag_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    
    chunks = meta["chunks"]
    print(f"\n文档块总数：{len(chunks)}")
    
    # 预先计算所有 chunk 的 embedding（节省时间）
    print("\n预计算文档块 embedding...")
    chunk_embs = []
    for i, c in enumerate(chunks):
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(chunks)}")
        chunk_embs.append(embed(c))
    chunk_embs = np.array(chunk_embs)
    print("✅ 完成")
    
    results = []
    k_values = [1, 3, 5]
    
    for query, gold_answers, difficulty in TEST_QUERIES:
        print(f"\n查询：{query}")
        
        # 编码查询
        q_emb = embed(query)
        
        # 计算与所有 chunk 的相似度
        similarities = np.array([cosine_sim(q_emb, ce) for ce in chunk_embs])
        
        # 找出最相似的 top-K
        top_k_idx = np.argsort(similarities)[-max(k_values):][::-1]
        
        # 计算黄金上下文的平均 embedding
        gold_embs = [embed(ga) for ga in gold_answers]
        avg_gold = np.mean(gold_embs, axis=0)
        
        # 确定哪些 chunk 是相关的（与黄金上下文相似度高）
        relevance = np.array([cosine_sim(ce, avg_gold) for ce in chunk_embs])
        threshold = 0.6
        relevant_mask = relevance >= threshold
        total_relevant = np.sum(relevant_mask)
        
        # 评估指标
        hits = [0, 0, 0]
        precisions = []
        recalls = []
        ndcgs = []
        
        for i, k in enumerate(k_values):
            top_k = top_k_idx[:k]
            rel_in_top = sum(1 for idx in top_k if relevant_mask[idx])
            
            hits[i] = 1 if rel_in_top > 0 else 0
            precisions.append(rel_in_top / k)
            recalls.append(rel_in_top / total_relevant if total_relevant > 0 else 0)
            
            # NDCG
            dcg = sum(relevance[idx] / np.log2(j+2) for j, idx in enumerate(top_k))
            ideal = sorted(relevance, reverse=True)[:k]
            idcg = sum(r / np.log2(j+2) for j, r in enumerate(ideal))
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
        
        # MRR
        first_rel = None
        for i, idx in enumerate(top_k_idx, 1):
            if relevant_mask[idx]:
                first_rel = i
                break
        mrr = 1/first_rel if first_rel else 0
        
        result = {
            "query": query,
            "difficulty": difficulty,
            "hit_rate": hits,
            "precision": precisions,
            "recall": recalls,
            "ndcg": ndcgs,
            "mrr": mrr,
            "top_3_indices": top_k_idx[:3].tolist(),
            "top_3_scores": similarities[top_k_idx[:3]].tolist()
        }
        results.append(result)
        
        print(f"  Hit@K: {[f'{h:.2f}' for h in hits]}")
        print(f"  P@K:   {[f'{p:.3f}' for p in precisions]}")
        print(f"  R@K:   {[f'{r:.3f}' for r in recalls]}")
        print(f"  N@K:   {[f'{n:.3f}' for n in ndcgs]}")
        print(f"  MRR:   {mrr:.3f}")
    
    # 聚合统计
    agg = {
        "total_queries": len(results),
        "avg_hit": [statistics.mean([r["hit_rate"][i] for r in results]) for i in range(len(k_values))],
        "avg_precision": [statistics.mean([r["precision"][i] for r in results]) for i in range(len(k_values))],
        "avg_recall": [statistics.mean([r["recall"][i] for r in results]) for i in range(len(k_values))],
        "avg_ndcg": [statistics.mean([r["ndcg"][i] for r in results]) for i in range(len(k_values))],
        "avg_mrr": statistics.mean([r["mrr"] for r in results]),
        "by_difficulty": {}
    }
    
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            agg["by_difficulty"][diff] = statistics.mean([r["hit_rate"][-1] for r in diff_results])
    
    # 输出摘要
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    print(f"平均 Hit@K: {[f'{h:.2f}' for h in agg['avg_hit']]}")
    print(f"平均 P@K:   {[f'{p:.3f}' for p in agg['avg_precision']]}")
    print(f"平均 R@K:   {[f'{r:.3f}' for r in agg['avg_recall']]}")
    print(f"平均 N@K:   {[f'{n:.3f}' for n in agg['avg_ndcg']]}")
    print(f"平均 MRR:   {agg['avg_mrr']:.3f}")
    print(f"\n按难度 (K=5):")
    for diff, hit in agg["by_difficulty"].items():
        print(f"  {diff}: {hit:.3f}")
    
    # 保存
    out_dir = Path("app/graphrag/evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(out_dir / f"quick_eval_{ts}.json", "w") as f:
        json.dump({"summary": agg, "details": results}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 已保存到 app/graphrag/evaluation/results/")
    
    return agg

if __name__ == "__main__":
    main()
