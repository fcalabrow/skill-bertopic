# skill-bertopic

Evaluación de modelos de embeddings sobre 1.000 tweets en español del dataset [`cardiffnlp/tweet_topic_multilingual`](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multilingual) (`train + validation + test`).

## Modelos evaluados

- `hotchpotch/bekko-embedding-v1-a8m`
- `hotchpotch/bekko-embedding-v1-a25m`
- `codefuse-ai/F2LLM-v2-160M`
- `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- `jinaai/jina-embeddings-v5-text-nano` (`default_task="clustering"`)

## Uso

```bash
uv venv
source .venv/bin/activate
uv pip install bertopic datasets peft
python evaluate_bertopic.py
python evaluate_kmeans.py
```

## Resultados

### BERTopic (parámetros por defecto)

| Modelo | Tamaño | Año | ms/oración | Topics | Outliers | ARI (multi) | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `jinaai/jina-embeddings-v5-text-nano` | 238.9M | 2026 | 100.08 | 18 | 199 | **0.2802** | **0.4719** |
| `hotchpotch/bekko-embedding-v1-a25m` | 123.2M | 2026 | 14.74 | 9 | 381 | 0.2794 | 0.3719 |
| `hotchpotch/bekko-embedding-v1-a8m` | 106.0M | 2026 | **7.07** | 9 | 414 | 0.2397 | 0.3646 |
| `codefuse-ai/F2LLM-v2-160M` | 159.2M | 2026 | 54.73 | 9 | 451 | 0.1030 | 0.2006 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 109.5M | 2022 | 42.72 | 2 | 182 | 0.0194 | 0.0161 |

### K-Means directo (`k = 19`, sin UMAP ni HDBSCAN)

| Modelo | Tamaño | Año | ms/oración | ARI | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `jinaai/jina-embeddings-v5-text-nano` | 238.9M | 2026 | 94.16 | 0.1676 | **0.4449** |
| `hotchpotch/bekko-embedding-v1-a25m` | 123.2M | 2026 | 14.87 | 0.1437 | 0.2899 |
| `hotchpotch/bekko-embedding-v1-a8m` | 106.0M | 2026 | **7.15** | **0.2273** | 0.2796 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 109.5M | 2022 | 39.75 | 0.1057 | 0.2230 |
| `codefuse-ai/F2LLM-v2-160M` | 159.2M | 2026 | 52.27 | 0.0590 | 0.2276 |

## Referencias

- [BERTopic - Vectorizers](https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html)
- [MTEB Leaderboard (Spanish v1 - Performance vs Size)](https://leaderboard.mteb.org/benchmark/MTEB(spa%2C%20v1)?tab=perf_size)
