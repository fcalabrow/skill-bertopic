---
name: bertopic-explorer
description: Exploratory topic modeling and clustering optimized for short Spanish text (typically tweets) in .txt, .csv, or .parquet datasets using BERTopic without loading full datasets into context.
compatibility: Requires Python 3.11+, uv, pandas, bertopic, sentence-transformers, peft, and pyarrow.
---

# BERTopic Explorer

Use this skill to discover topics autonomously in large text or tweet datasets (`.txt`, `.csv`, `.parquet`) without loading raw datasets into your context window.

## Workflow

1. **Run exploratory clustering** (use `--sample` on large datasets):
   ```bash
   uv run scripts/topic_modeling.py run \
     --input data.parquet \
     --text-col text \
     --id-col id \
     --sample 2000 \
     --out-dir ./topic_run
   ```
   - Supports `.txt` (one document per line), `.csv`, and `.parquet`.
   - Automatically cleans tweets (URLs, `RT @user:`, `@mentions`) before vectorization.
   - Automatically sets `min_topic_size = max(5, min(50, int(0.015 * N)))`.
   - **Speed vs. precision (`--slow`)**:
     - Default mode uses a fast, lightweight embedding model.
     - Pass `--slow` to use a heavier, more accurate embedding model for finer clustering (~6.8x slower).
     - Optional: run `uv run scripts/topic_modeling.py estimate --n 2000 [--slow]` (or `--input data.parquet`) to benchmark vectorization speed on the current hardware before running on large datasets.
   - Outputs:
     - `topic_run/topics_summary.json`: compact overview (`topic_id`, `count`, `pct`, `keywords` with Spanish stopwords filtered out, `representative_docs`). Read this first.
     - `topic_run/clusters.parquet`: `id`, `topic`, `topic_prob` (no duplicated text).

2. **Inspect specific topics**:
   ```bash
   uv run scripts/topic_modeling.py inspect \
     --clusters ./topic_run/clusters.parquet \
     --input data.parquet \
     --text-col text \
     --id-col id \
     --topic 0 \
     --n 10
   ```

3. **Deep analysis with `pandas`**:
   If you need custom aggregations or filtering, load `clusters.parquet` with `pandas` and join with the original dataset on `id`.

## Interpreting Results

- `topic == -1`: Outliers (documents that HDBSCAN did not assign to any dense cluster). In noisy Twitter data, 20–40% outliers is normal.
- `topics_summary.json` gives you keywords and representative tweets per topic so you can name and summarize topics without reading the full dataset.
