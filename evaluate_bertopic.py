import json
import os
import time
from collections import Counter

from bertopic import BERTopic
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)

MODELS = [
    "hotchpotch/bekko-embedding-v1-a8m",
    "hotchpotch/bekko-embedding-v1-a25m",
    "codefuse-ai/F2LLM-v2-160M",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "jinaai/jina-embeddings-v5-text-nano",
]

RESULTS_JSON = "results.json"


def load_spanish_tweets():
    ds = load_dataset("cardiffnlp/tweet_topic_multilingual", "es")
    full_ds = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    texts = full_ds["text"]
    multi_labels = [
        [idx for idx, val in enumerate(row) if val == 1]
        for row in full_ds["label"]
    ]
    label_sets = full_ds["label_name_flatten"]
    return texts, multi_labels, label_sets


def optimistic_single_labels(multi_labels, pred_topics):
    cluster_label_counts = {}
    for labels, cluster in zip(multi_labels, pred_topics):
        if cluster not in cluster_label_counts:
            cluster_label_counts[cluster] = Counter()
        for lbl in labels:
            cluster_label_counts[cluster][lbl] += 1

    resolved = []
    for labels, cluster in zip(multi_labels, pred_topics):
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
    texts, multi_labels, label_sets = load_spanish_tweets()
    print(f"Loaded {len(texts)} Spanish tweets.\n")

    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            results_map = json.load(f)
    else:
        results_map = {}

    api = HfApi()

    for model_name in MODELS:
        print(f"=== Processing: {model_name} ===")
        entry = results_map.get(model_name, {})
        if "nmi" in entry and "v_measure" in entry and "ms_per_sent" in entry:
            print("Using cached results.")
            print(
                f"Size: {entry['size']} | Year: {entry['year']} | Speed: {entry['ms_per_sent']:.2f} ms/sent | "
                f"Topics: {entry['topics']} | Outliers: {entry['outliers']}/{len(texts)} | "
                f"ARI (multi): {entry['ari_multilabel']:.4f} | NMI: {entry['nmi']:.4f} | "
                f"V-measure: {entry['v_measure']:.4f}\n"
            )
            continue

        kwargs = {"trust_remote_code": True}
        if "jina-embeddings-v5" in model_name:
            kwargs["model_kwargs"] = {"default_task": "clustering"}
        embedding_model = SentenceTransformer(model_name, **kwargs)
        size_str, year = get_hf_metadata(api, model_name, embedding_model)

        # Warmup + single encode pass reused for BERTopic
        embedding_model.encode(texts[:5], show_progress_bar=False)
        t0 = time.perf_counter()
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        ms_per_sent = ((time.perf_counter() - t0) / len(texts)) * 1000.0

        topic_model = BERTopic(embedding_model=embedding_model)
        topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
        n_topics = len(set(topics) - {-1})
        n_outliers = sum(1 for t in topics if t == -1)
        resolved_labels = optimistic_single_labels(multi_labels, topics)

        ari_multilabel = adjusted_rand_score(resolved_labels, topics)
        ari_labelset = adjusted_rand_score(label_sets, topics)
        nmi = normalized_mutual_info_score(resolved_labels, topics)
        v_meas = v_measure_score(resolved_labels, topics)

        entry = {
            "size": size_str,
            "year": year,
            "ms_per_sent": ms_per_sent,
            "topics": n_topics,
            "outliers": n_outliers,
            "ari_multilabel": ari_multilabel,
            "ari_labelset": ari_labelset,
            "nmi": nmi,
            "v_measure": v_meas,
        }
        results_map[model_name] = entry

        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results_map, f, indent=2)

        print(
            f"Size: {size_str} | Year: {year} | Speed: {ms_per_sent:.2f} ms/sent | "
            f"Topics: {n_topics} | Outliers: {n_outliers}/{len(texts)} | "
            f"ARI (multi): {ari_multilabel:.4f} | NMI: {nmi:.4f} | V-measure: {v_meas:.4f}\n"
        )

    header = (
        f"{'Model':<48} | {'Size':>6} | {'Year':>4} | {'ms/sent':>7} | "
        f"{'Topics':>6} | {'Outliers':>8} | {'ARI (multi)':>11} | {'NMI':>6} | {'V-meas':>6}"
    )
    sep = "-" * len(header)
    lines = ["=== Summary ===", header, sep]
    for m in MODELS:
        r = results_map[m]
        lines.append(
            f"{m:<48} | {r['size']:>6} | {r['year']:>4} | {r['ms_per_sent']:>7.2f} | "
            f"{r['topics']:>6} | {r['outliers']:>8} | "
            f"{r['ari_multilabel']:>11.4f} | {r['nmi']:>6.4f} | {r['v_measure']:>6.4f}"
        )

    summary_text = "\n".join(lines)
    print(summary_text)
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")


if __name__ == "__main__":
    main()
