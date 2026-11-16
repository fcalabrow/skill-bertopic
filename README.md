# skill-bertopic

Evaluación de modelos de embeddings con BERTopic (parámetros por defecto) sobre tweets en español del dataset [`cardiffnlp/tweet_topic_multilingual`](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multilingual).

## Modelos evaluados

- `hotchpotch/bekko-embedding-v1-a8m`
- `codefuse-ai/F2LLM-v2-160M`
- `sentence-transformers/multi-qa-mpnet-base-dot-v1`

## Uso

```bash
uv venv
source .venv/bin/activate
uv pip install bertopic datasets
python evaluate_bertopic.py
```
