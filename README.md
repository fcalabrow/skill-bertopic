# bertopic-explorer

Skill para descubrimiento exploratorio de tópicos con **BERTopic** y `hotchpotch/bekko-embedding-v1-a25m`, diseñado para que un agente autónomo explore datasets grandes (`.txt`, `.csv`, `.parquet`) sin saturar su ventana de contexto.

> Los experimentos de evaluación de modelos están en la rama [`experiments`](../../tree/experiments).

## Uso rápido

```bash
# 1. Ejecutar clustering exploratorio sobre una muestra
uv run scripts/topic_modeling.py run \
  --input tweets.parquet --text-col text --id-col id \
  --sample 2000 --out-dir ./topic_run

# 2. Inspeccionar ejemplos de un tópico
uv run scripts/topic_modeling.py inspect \
  --clusters ./topic_run/clusters.parquet \
  --input tweets.parquet --text-col text --id-col id \
  --topic 0 --n 10
```

## Tests unitarios

```bash
python -m unittest discover -s tests -v
```
