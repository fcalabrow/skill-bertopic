# skill-bertopic

Evaluación de modelos de embeddings en español sobre tres datasets públicos usando **BERTopic (parámetros por defecto)** y **K-Means directo** (`k = cantidad de categorías del dataset`).

## Modelos evaluados

- `hotchpotch/bekko-embedding-v1-a8m`
- `hotchpotch/bekko-embedding-v1-a25m`
- `codefuse-ai/F2LLM-v2-160M`
- `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- `jinaai/jina-embeddings-v5-text-nano` (`default_task="clustering"`)
- `sentence-transformers/static-similarity-mrl-multilingual-v1` (baseline de embeddings estáticos)

## Uso

```bash
uv venv
source .venv/bin/activate
uv pip install bertopic datasets peft
python evaluate_bertopic.py
python evaluate_kmeans.py
python evaluate_datasets.py --dataset sib200
python evaluate_datasets.py --dataset massive
```

## Resultados

### 1. `cardiffnlp/tweet_topic_multilingual` (`es`, 1.000 tweets, multilabel)

#### BERTopic (parámetros por defecto)

| Modelo | Tamaño | Año | ms/oración | Topics | Outliers | ARI (multi) | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `jinaai/jina-embeddings-v5-text-nano` | 238.9M | 2026 | 100.08 | 18 | 199 | **0.2802** | **0.4719** |
| `hotchpotch/bekko-embedding-v1-a25m` | 123.2M | 2026 | 14.74 | 9 | 381 | 0.2794 | 0.3719 |
| `hotchpotch/bekko-embedding-v1-a8m` | 106.0M | 2026 | 7.07 | 9 | 414 | 0.2397 | 0.3646 |
| `codefuse-ai/F2LLM-v2-160M` | 159.2M | 2026 | 54.73 | 9 | 451 | 0.1030 | 0.2006 |
| `sentence-transformers/static-similarity-mrl-multilingual-v1` | 108.4M | 2024 | **0.10** | 16 | 575 | 0.0101 | 0.1217 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 109.5M | 2022 | 42.72 | 2 | 182 | 0.0194 | 0.0161 |

#### K-Means directo (`k = 19`)

| Modelo | Tamaño | Año | ms/oración | ARI | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `jinaai/jina-embeddings-v5-text-nano` | 238.9M | 2026 | 94.16 | 0.1676 | **0.4449** |
| `hotchpotch/bekko-embedding-v1-a25m` | 123.2M | 2026 | 14.87 | 0.1437 | 0.2899 |
| `hotchpotch/bekko-embedding-v1-a8m` | 106.0M | 2026 | 7.15 | **0.2273** | 0.2796 |
| `codefuse-ai/F2LLM-v2-160M` | 159.2M | 2026 | 52.27 | 0.0590 | 0.2276 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 109.5M | 2022 | 39.75 | 0.1057 | 0.2230 |
| `sentence-transformers/static-similarity-mrl-multilingual-v1` | 108.4M | 2024 | **0.10** | 0.0922 | 0.1610 |

---

### 2. `Davlan/sib200` (`spa_Latn`, 1.004 docs, `k = 7`, label `category`)

#### BERTopic (parámetros por defecto)

| Modelo | Tamaño | Año | ms/oración | Topics | Outliers | ARI | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `hotchpotch/bekko-embedding-v1-a8m` | 106.0M | 2026 | 6.14 | 18 | 236 | **0.3400** | **0.5330** |
| `jinaai/jina-embeddings-v5-text-nano` | 238.9M | 2026 | 74.11 | 23 | 61 | 0.2825 | 0.5223 |
| `codefuse-ai/F2LLM-v2-160M` | 159.2M | 2026 | 53.70 | 14 | 346 | 0.3169 | 0.5018 |
| `hotchpotch/bekko-embedding-v1-a25m` | 123.2M | 2026 | 12.85 | 23 | 293 | 0.1682 | 0.4510 |
| `sentence-transformers/static-similarity-mrl-multilingual-v1` | 108.4M | 2024 | **0.10** | 16 | 414 | 0.1048 | 0.3287 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 109.5M | 2022 | 46.76 | 16 | 472 | 0.0562 | 0.2745 |

#### K-Means directo (`k = 7`)

| Modelo | Tamaño | Año | ms/oración | ARI | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `jinaai/jina-embeddings-v5-text-nano` | 238.9M | 2026 | 74.11 | **0.4728** | **0.5799** |
| `codefuse-ai/F2LLM-v2-160M` | 159.2M | 2026 | 53.70 | 0.4330 | 0.4833 |
| `hotchpotch/bekko-embedding-v1-a25m` | 123.2M | 2026 | 12.85 | 0.4519 | 0.4591 |
| `hotchpotch/bekko-embedding-v1-a8m` | 106.0M | 2026 | 6.14 | 0.3429 | 0.3943 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 109.5M | 2022 | 46.76 | 0.1146 | 0.1946 |
| `sentence-transformers/static-similarity-mrl-multilingual-v1` | 108.4M | 2024 | **0.10** | 0.1344 | 0.1896 |

---

### 3. `AmazonScience/massive` (`es-ES`, validation, 1.000 docs, `k = 16`, label `scenario`)

#### BERTopic (parámetros por defecto)

| Modelo | Tamaño | Año | ms/oración | Topics | Outliers | ARI | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `codefuse-ai/F2LLM-v2-160M` | 159.2M | 2026 | 13.88 | 23 | 42 | **0.5485** | **0.7118** |
| `hotchpotch/bekko-embedding-v1-a25m` | 123.2M | 2026 | 3.95 | 21 | 153 | 0.4315 | 0.6477 |
| `jinaai/jina-embeddings-v5-text-nano` | 238.9M | 2026 | 21.44 | 22 | 70 | 0.4456 | 0.6455 |
| `hotchpotch/bekko-embedding-v1-a8m` | 106.0M | 2026 | 1.79 | 23 | 182 | 0.3721 | 0.6223 |
| `sentence-transformers/static-similarity-mrl-multilingual-v1` | 108.4M | 2024 | **0.04** | 21 | 146 | 0.3163 | 0.5467 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 109.5M | 2022 | 14.09 | 23 | 216 | 0.2321 | 0.5206 |

#### K-Means directo (`k = 16`)

| Modelo | Tamaño | Año | ms/oración | ARI | V-measure |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `codefuse-ai/F2LLM-v2-160M` | 159.2M | 2026 | 13.88 | **0.5296** | **0.7067** |
| `jinaai/jina-embeddings-v5-text-nano` | 238.9M | 2026 | 21.44 | 0.4577 | 0.6381 |
| `hotchpotch/bekko-embedding-v1-a25m` | 123.2M | 2026 | 3.95 | 0.3938 | 0.6141 |
| `hotchpotch/bekko-embedding-v1-a8m` | 106.0M | 2026 | 1.79 | 0.2141 | 0.4742 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 109.5M | 2022 | 14.09 | 0.2150 | 0.4308 |
| `sentence-transformers/static-similarity-mrl-multilingual-v1` | 108.4M | 2024 | **0.04** | 0.1392 | 0.3987 |

## Referencias

- [BERTopic - Vectorizers](https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html)
- [MTEB Leaderboard (Spanish v1 - Performance vs Size)](https://leaderboard.mteb.org/benchmark/MTEB(spa%2C%20v1)?tab=perf_size)
