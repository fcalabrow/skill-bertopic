import json
import time
from collections import Counter

from bertopic import BERTopic
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, v_measure_score

MODEL_NAME = "sentence-transformers/static-similarity-mrl-multilingual-v1"
ALL_MODELS = [
    "hotchpotch/bekko-embedding-v1-a8m",
    "hotchpotch/bekko-embedding-v1-a25m",
    "codefuse-ai/F2LLM-v2-160M",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "jinaai/jina-embeddings-v5-text-nano",
    MODEL_NAME,
]


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
    size_str = f"{num_params / 1e9:.2f}B" if num_params >= 1e9 else f"{num_params / 1e6:.1f}M"
    return size_str, year


def encode_and_time(emb, texts):
    emb.encode(texts[:5], show_progress_bar=False)
    t0 = time.perf_counter()
    embeddings = emb.encode(texts, show_progress_bar=False)
    ms_per_sent = ((time.perf_counter() - t0) / len(texts)) * 1000.0
    return embeddings, ms_per_sent


def main():
    api = HfApi()
    emb = SentenceTransformer(MODEL_NAME)
    size_str, year = get_hf_metadata(api, MODEL_NAME, emb)
    print(f"Loaded {MODEL_NAME} | Size: {size_str} | Year: {year}")

    # 1. Tweet Topic Multilingual (es)
    ds_tt = load_dataset("cardiffnlp/tweet_topic_multilingual", "es")
    full_tt = concatenate_datasets([ds_tt["train"], ds_tt["validation"], ds_tt["test"]])
    tt_texts = full_tt["text"]
    tt_multi = [[idx for idx, val in enumerate(row) if val == 1] for row in full_tt["label"]]
    tt_sets = full_tt["label_name_flatten"]
    tt_k = len(ds_tt["train"].features["label"].feature.names)

    tt_emb, tt_ms = encode_and_time(emb, tt_texts)
    topics, _ = BERTopic(embedding_model=emb).fit_transform(tt_texts, embeddings=tt_emb)
    tt_res = optimistic_single_labels(tt_multi, topics)

    with open("results.json", "r", encoding="utf-8") as f:
        r_bt = json.load(f)
    r_bt[MODEL_NAME] = {
        "size": size_str,
        "year": year,
        "ms_per_sent": tt_ms,
        "topics": len(set(topics) - {-1}),
        "outliers": sum(1 for t in topics if t == -1),
        "ari_multilabel": adjusted_rand_score(tt_res, topics),
        "ari_labelset": adjusted_rand_score(tt_sets, topics),
        "v_measure": v_measure_score(tt_res, topics),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(r_bt, f, indent=2)

    clusters = KMeans(n_clusters=tt_k, random_state=42, n_init="auto").fit_predict(tt_emb)
    tt_res_km = optimistic_single_labels(tt_multi, clusters)
    with open("results_kmeans.json", "r", encoding="utf-8") as f:
        r_km = json.load(f)
    r_km[MODEL_NAME] = {
        "size": size_str,
        "year": year,
        "ms_per_sent": tt_ms,
        "k": tt_k,
        "ari": adjusted_rand_score(tt_res_km, clusters),
        "v_measure": v_measure_score(tt_res_km, clusters),
    }
    with open("results_kmeans.json", "w", encoding="utf-8") as f:
        json.dump(r_km, f, indent=2)

    # Regenerate results.txt and results_kmeans.txt
    h_bt = f"{'Model':<56} | {'Size':>6} | {'Year':>4} | {'ms/sent':>7} | {'Topics':>6} | {'Outliers':>8} | {'ARI (multi)':>11} | {'V-meas':>6}"
    lines_bt = ["=== Summary ===", h_bt, "-" * len(h_bt)]
    for m in ALL_MODELS:
        r = r_bt[m]
        lines_bt.append(f"{m:<56} | {r['size']:>6} | {r['year']:>4} | {r['ms_per_sent']:>7.2f} | {r['topics']:>6} | {r['outliers']:>8} | {r['ari_multilabel']:>11.4f} | {r['v_measure']:>6.4f}")
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines_bt) + "\n")

    h_km = f"{'Model':<56} | {'Size':>6} | {'Year':>4} | {'ms/sent':>7} | {'k':>2} | {'ARI':>6} | {'V-meas':>6}"
    lines_km = ["=== K-Means Summary ===", h_km, "-" * len(h_km)]
    for m in ALL_MODELS:
        r = r_km[m]
        lines_km.append(f"{m:<56} | {r['size']:>6} | {r['year']:>4} | {r['ms_per_sent']:>7.2f} | {r['k']:>2} | {r['ari']:>6.4f} | {r['v_measure']:>6.4f}")
    with open("results_kmeans.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines_km) + "\n")

    # 2. SIB200 & MASSIVE
    datasets_cfg = [
        (
            "SIB200 (spa_Latn)",
            "results_sib200.json",
            "results_sib200.txt",
            lambda: (
                lambda d: (d["text"], d["category"])
            )(concatenate_datasets(list(load_dataset("Davlan/sib200", "spa_Latn").values()))),
        ),
        (
            "MASSIVE (es-ES, validation, 1000 docs)",
            "results_massive.json",
            "results_massive.txt",
            lambda: (
                lambda d: (d["utt"], d["scenario"])
            )(
                load_dataset("AmazonScience/massive", revision="refs/convert/parquet", split="validation")
                .filter(lambda x: x["locale"].startswith("es"))
                .select(range(1000))
            ),
        ),
    ]

    for title, jpath, tpath, loader in datasets_cfg:
        texts, labels = loader()
        k = len(set(labels))
        embs, ms = encode_and_time(emb, texts)
        topics, _ = BERTopic(embedding_model=emb).fit_transform(texts, embeddings=embs)
        clusters = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embs)

        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["bertopic"][MODEL_NAME] = {
            "size": size_str,
            "year": year,
            "ms_per_sent": ms,
            "topics": len(set(topics) - {-1}),
            "outliers": sum(1 for t in topics if t == -1),
            "ari": adjusted_rand_score(labels, topics),
            "v_measure": v_measure_score(labels, topics),
        }
        data["kmeans"][MODEL_NAME] = {
            "size": size_str,
            "year": year,
            "ms_per_sent": ms,
            "k": k,
            "ari": adjusted_rand_score(labels, clusters),
            "v_measure": v_measure_score(labels, clusters),
        }
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        h1 = f"{'Model':<56} | {'Size':>6} | {'Year':>4} | {'ms/sent':>7} | {'Topics':>6} | {'Outliers':>8} | {'ARI':>6} | {'V-meas':>6}"
        out = [f"=== {title} - BERTopic ===", h1, "-" * len(h1)]
        for m in ALL_MODELS:
            r = data["bertopic"][m]
            out.append(f"{m:<56} | {r['size']:>6} | {r['year']:>4} | {r['ms_per_sent']:>7.2f} | {r['topics']:>6} | {r['outliers']:>8} | {r['ari']:>6.4f} | {r['v_measure']:>6.4f}")
        h2 = f"{'Model':<56} | {'Size':>6} | {'Year':>4} | {'ms/sent':>7} | {'k':>2} | {'ARI':>6} | {'V-meas':>6}"
        out.extend([f"\n=== {title} - K-Means ===", h2, "-" * len(h2)])
        for m in ALL_MODELS:
            r = data["kmeans"][m]
            out.append(f"{m:<56} | {r['size']:>6} | {r['year']:>4} | {r['ms_per_sent']:>7.2f} | {r['k']:>2} | {r['ari']:>6.4f} | {r['v_measure']:>6.4f}")
        with open(tpath, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")

    print("All reports regenerated successfully!")


if __name__ == "__main__":
    main()
