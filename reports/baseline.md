# Evaluation report

Generated 2026-11-16T09:52:36Z - 14 papers, 899 chunks, 50 golden questions (40 answerable, 10 unanswerable). Generator: `extractive-reader-v1`. Run time 32.8s.

## 1. Retrieval ablation

Share of answerable questions whose answer-bearing passage is retrieved. Reranker weights were tuned on the dev half only; **test** is held out.

| Mode | Split | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| `bm25` | test | 0.632 | 0.737 | 0.895 | 1.000 | 0.734 | 0.663 |
| `bm25` | dev | 0.571 | 0.714 | 0.857 | 0.952 | 0.692 | 0.576 |
| `dense` | test | 0.474 | 0.737 | 0.895 | 0.947 | 0.623 | 0.569 |
| `dense` | dev | 0.429 | 0.619 | 0.809 | 0.905 | 0.564 | 0.533 |
| `hybrid` | test | 0.632 | 0.789 | 0.895 | 1.000 | 0.751 | 0.680 |
| `hybrid` | dev | 0.476 | 0.667 | 0.905 | 0.952 | 0.628 | 0.570 |
| `hybrid+rerank` | test | 0.632 | 0.842 | 0.947 | 1.000 | 0.744 | 0.668 |
| `hybrid+rerank` | dev | 0.524 | 0.809 | 0.857 | 0.952 | 0.664 | 0.580 |

## 2. Answer quality

| Split | n | Answer accuracy | Wrong answers | False refusals | Correct refusals | Citation precision | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 24 | 63.2% | 26.3% | 10.5% | 100.0% | 100.0% | 16.8 ms | 19.5 ms |
| dev | 26 | 57.1% | 38.1% | 4.8% | 100.0% | 95.8% | 17.4 ms | 19.3 ms |
| all | 50 | 60.0% | 32.5% | 7.5% | 100.0% | 97.8% | 16.9 ms | 19.5 ms |

## 3. Resilience under injected faults

Every answerable question, three seeded rounds per cell. *Availability* = the user got an answer or a deliberate refusal rather than an error. *Poisoned* = a fabricated or uncited claim reached the user.

| Fault profile | Variant | Availability | Correct | Poisoned served | Fallback rate | Attempts |
|---|---|---:|---:|---:|---:|---:|
| flaky-network | naive | 52.5% | 31.7% | 0.0% | 0.0% | 1.0 |
| flaky-network | validation only | 52.5% | 31.7% | 0.0% | 0.0% | 1.0 |
| flaky-network | resilient | 100.0% | 60.0% | 0.0% | 69.2% | 1.85 |
| bad-outputs | naive | 69.2% | 40.8% | 22.5% | 0.0% | 1.0 |
| bad-outputs | validation only | 68.3% | 35.8% | 0.0% | 0.0% | 1.0 |
| bad-outputs | resilient | 100.0% | 56.7% | 0.0% | 60.0% | 1.74 |
| everything | naive | 54.2% | 30.8% | 7.5% | 0.0% | 1.0 |
| everything | validation only | 54.2% | 30.0% | 0.0% | 0.0% | 1.0 |
| everything | resilient | 100.0% | 58.3% | 0.0% | 65.0% | 1.85 |
| outage | naive | 0.0% | 0.0% | 0.0% | 0.0% | 1.0 |
| outage | validation only | 0.0% | 0.0% | 0.0% | 0.0% | 1.0 |
| outage | resilient | 100.0% | 60.0% | 0.0% | 100.0% | 2.05 |

## Misses

Answerable questions the system did not answer correctly:

- `tf-warmup` (dev) - answered
- `bert-mask-rate` (test) - answered
- `bert-pretraining-data` (dev) - answered
- `dpr-corpus-size` (test) - abstained
- `colbert-speed` (test) - answered
- `rag-formulations` (test) - answered
- `rag-retriever` (dev) - answered
- `beir-datasets` (dev) - answered
- `beir-metric` (dev) - answered
- `lora-init` (test) - answered
- `react-decision-gains` (test) - abstained
- `hyde-encoder` (dev) - answered
- `hyde-name` (dev) - answered
- `lim-curve` (dev) - abstained
- `lim-best-position` (dev) - answered
- `ragas-dimensions` (test) - answered
