import json
import time
import weaviate

from sentence_transformers import SentenceTransformer
from rerank import rerank_search
from retrieval_helpers import hybrid_search

client = weaviate.Client("http://localhost:8080")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

rows = []
with open("data/retrieval_eval.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))

hits = 0
rr_sum = 0

hybrid_times = []
total_times = []

for row in rows:
    query = row["query"]
    gold = row["gold_doc_id"]

    t0 = time.perf_counter()
    results = rerank_search(client, query, embedder, k_in=50, k_out=5)
    total_ms = (time.perf_counter() - t0) * 1000

    total_times.append(total_ms)

    if gold in results:
        hits += 1
        rank = results.index(gold) + 1
        rr_sum += 1 / rank

recall5 = hits / len(rows)
mrr = rr_sum / len(rows)

print("Recall@5 =", recall5)
print("MRR =", mrr)
print("Avg Total Latency (ms) =", sum(total_times) / len(total_times))