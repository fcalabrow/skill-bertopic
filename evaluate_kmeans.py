import json
import time
from collections import Counter

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    v_measure_score,
)

MODELS = [
    "hotchpotch/bekko-embedding-v1-a8m",
    "hotchpotch/bekko-embedding-v1-a25m",
    "codefuse-ai/F2LLM-v2-160M",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "jinaai/jina-embeddings-v5-text-nano",
]

RESULTS_JSON = "results_kmeans.json"


def load_spanish_tweets():
    ds = load_dataset("cardiffnlp/tweet_topic_multilingual", "es")
    full_ds = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    texts = full_ds["text"]
    num_classes = len(ds["train"].features["label"].feature.names)
    multi_labels = [
        [idx for idx, val in enumerate(row) if val == 1]
        for row in full_ds["label"]
    ]
    return texts, multi_labels, num_classes


def optimistic_single_labels(multi_labels, pred_clusters):
    cluster_label_counts = {}
    for labels, cluster in zip(multi_labels, pred_clusters):
        if cluster not in cluster_label_counts:
            cluster_label_counts[cluster] = Counter()
        for lbl in labels:
            cluster_label_counts[cluster][lbl] += 1

    resolved = []
    for labels, cluster in zip(multi_labels, pred_clusters):
        if not labels:
            resolved.append(-1)
            continue
        counts = cluster_label_counts[cluster]
        best_label = max(labels, key=lambda l: counts[l])
        resolved.append(best_label)
    return resolved


def get_hf_metadata(api, model_name, embedding_model):
    year = "N/A"
    try:
        info = api.model_info(model_name)
        if info.created_at:
            year = str(info.created_at.year)
    except Exception:
        pass

    num_params = sum(p.numel() for p in embedding_model.parameters())
    if num_params >= 1e9:
        size_str = f"{num_params / 1e9:.2f}B"
    else:
        size_str = f"{num_params / 1e6:.1f}M"
    return size_str, year


def main():
    texts, multi_labels, num_classes = load_spanish_tweets()
    print(f"Loaded {len(texts)} Spanish tweets. Dataset topics (k): {num_classes}\n")

    api = HfApi()
    results_map = {}

    for model_name in MODELS:
        print(f"=== Processing: {model_name} ===")
        kwargs = {"trust_remote_code": True}
        if "jina-embeddings-v5" in model_name:
            kwargs["model_kwargs"] = {"default_task": "clustering"}
        embedding_model = SentenceTransformer(model_name, **kwargs)
        size_str, year = get_hf_metadata(api, model_name, embedding_model)

        embedding_model.encode(texts[:5], show_progress_bar=False)
        t0 = time.perf_counter()
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        ms_per_sent = ((time.perf_counter() - t0) / len(texts)) * 1000.0

        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(embeddings)

        resolved_labels = optimistic_single_labels(multi_labels, clusters)
        ari = adjusted_rand_score(resolved_labels, clusters)
        v_meas = v_measure_score(resolved_labels, clusters)

        results_map[model_name] = {
            "size": size_str,
            "year": year,
            "ms_per_sent": ms_per_sent,
            "k": num_classes,
            "ari": ari,
            "v_measure": v_meas,
        }
        print(
            f"Size: {size_str} | Year: {year} | Speed: {ms_per_sent:.2f} ms/sent | "
            f"ARI: {ari:.4f} | V-measure: {v_meas:.4f}\n"
        )

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_map, f, indent=2)

    header = (
        f"{'Model':<48} | {'Size':>6} | {'Year':>4} | {'ms/sent':>7} | "
        f"{'k':>2} | {'ARI':>6} | {'V-meas':>6}"
    )
    sep = "-" * len(header)
    lines = ["=== K-Means Summary ===", header, sep]
    for m in MODELS:
        r = results_map[m]
        lines.append(
            f"{m:<48} | {r['size']:>6} | {r['year']:>4} | {r['ms_per_sent']:>7.2f} | "
            f"{r['k']:>2} | {r['ari']:>6.4f} | {r['v_measure']:>6.4f}"
        )

    summary_text = "\n".join(lines)
    print(summary_text)
    with open("results_kmeans.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")


if __name__ == "__main__":
    main()
