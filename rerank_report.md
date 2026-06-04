## Setup

- Hybrid k_in: 50  
- Re-ranked k_out: 5  
- Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2  
- Hardware (CPU model, RAM, OS): CPU-based local environment (Windows, single machine, no GPU acceleration)

## Metrics Table

| Pipeline | recall@5 | MRR | per-query latency (ms) |
|---|---|---|---|
| Hybrid (lab baseline) | 0.69 | 0.52 | ~50 ms |
| Hybrid + cross-encoder rerank | 0.783 | 0.624 | ~7919 ms |

Stage 1 (hybrid retrieval) is fast (~tens of ms), while stage 2 (cross-encoder scoring over 50 query–document pairs) dominates latency and increases total runtime significantly.

## When Does Re-Ranking Pay Off?

Re-ranking clearly improves ranking quality in cases where the hybrid retriever retrieves relevant documents but places them too low in the top-k results. For example, in several queries from the evaluation set, the correct document was ranked around positions 8–20 in the hybrid stage but moved into the top 3 after cross-encoder scoring. This explains the increase in both Recall@5 (0.69 → 0.783) and MRR (0.52 → 0.624).

In contrast, for “easy” queries where the hybrid retriever already ranks the correct document at position 1–3, the cross-encoder does not significantly change the output.

## Latency Overhead

The cross-encoder introduces the main computational cost. While hybrid retrieval alone operates in ~50 ms per query, adding reranking increases total latency to ~7.9 seconds. The overhead scales linearly with k_in, since the cross-encoder evaluates each query–document pair individually (O(k_in) forward passes per query). In this case, k_in = 50 leads to substantial slowdown.

Unlike hybrid retrieval, corpus size does not directly affect cross-encoder latency, but larger corpora typically require higher k_in to maintain recall, indirectly increasing cost.

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

Re-ranking becomes impractical in real-time or high-throughput systems. Assuming ~8 seconds per query, the system supports roughly ~0.1 QPS, which is far below production requirements (typically 10–1000 QPS). Therefore, this approach is only viable for low-traffic, high-precision applications.

At higher query volumes, the cross-encoder becomes the bottleneck. A lighter reranker, reduced k_in (e.g., 10–20), or cached retrieval results would be required. Alternatively, deploying a distilled cross-encoder or GPU acceleration would be necessary to make this architecture scalable.