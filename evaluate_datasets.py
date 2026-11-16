import json
import time

from bertopic import BERTopic
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, v_measure_score

MODELS = [
    "hotchpotch/bekko-embedding-v1-a8m",
    "hotchpotch/bekko-embedding-v1-a25m",
    "codefuse-ai/F2LLM-v2-160M",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "jinaai/jina-embeddings-v5-text-nano",
]


def load_massive_es():
    ds = load_dataset(
        "AmazonScience/massive",
        revision="refs/convert/parquet",
        split="validation",
    )
    es_ds = ds.filter(lambda x: x["locale"].startswith("es"))
    es_ds = es_ds.select(range(min(1000, len(es_ds))))
    texts = es_ds["utt"]
    labels = es_ds["scenario"]
    return texts, labels


def load_sib200_es():
    ds = load_dataset("Davlan/sib200", "spa_Latn")
    full_ds = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    texts = full_ds["text"]
    labels = full_ds["category"]
    return texts, labels


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


def evaluate_dataset(name, texts, labels, loaded_models, api):
    k = len(set(labels))
    print(f"\n================ {name} ({len(texts)} docs, k={k}) ================\n")

    bt_results = {}
    km_results = {}

    for model_name in MODELS:
        print(f"--- {model_name} ---")
        embedding_model, size_str, year = loaded_models[model_name]

        embedding_model.encode(texts[:5], show_progress_bar=False)
        t0 = time.perf_counter()
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        ms_per_sent = ((time.perf_counter() - t0) / len(texts)) * 1000.0

        # 1. BERTopic (default)
        topic_model = BERTopic(embedding_model=embedding_model)
        topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
        n_topics = len(set(topics) - {-1})
        n_outliers = sum(1 for t in topics if t == -1)
        bt_ari = adjusted_rand_score(labels, topics)
        bt_v = v_measure_score(labels, topics)

        bt_results[model_name] = {
            "size": size_str,
            "year": year,
            "ms_per_sent": ms_per_sent,
            "topics": n_topics,
            "outliers": n_outliers,
            "ari": bt_ari,
            "v_measure": bt_v,
        }

        # 2. K-Means (k = num_classes)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(embeddings)
        km_ari = adjusted_rand_score(labels, clusters)
        km_v = v_measure_score(labels, clusters)

        km_results[model_name] = {
            "size": size_str,
            "year": year,
            "ms_per_sent": ms_per_sent,
            "k": k,
            "ari": km_ari,
            "v_measure": km_v,
        }

        print(
            f"Speed: {ms_per_sent:.2f} ms/sent | "
            f"BERTopic -> Topics: {n_topics}, Outliers: {n_outliers}, ARI: {bt_ari:.4f}, V-meas: {bt_v:.4f} | "
            f"KMeans -> ARI: {km_ari:.4f}, V-meas: {km_v:.4f}"
        )

    return bt_results, km_results


def format_tables(name, bt_results, km_results):
    lines = [f"=== {name} - BERTopic ==="]
    h1 = (
        f"{'Model':<48} | {'Size':>6} | {'Year':>4} | {'ms/sent':>7} | "
        f"{'Topics':>6} | {'Outliers':>8} | {'ARI':>6} | {'V-meas':>6}"
    )
    lines.extend([h1, "-" * len(h1)])
    for m in MODELS:
        r = bt_results[m]
        lines.append(
            f"{m:<48} | {r['size']:>6} | {r['year']:>4} | {r['ms_per_sent']:>7.2f} | "
            f"{r['topics']:>6} | {r['outliers']:>8} | {r['ari']:>6.4f} | {r['v_measure']:>6.4f}"
        )

    lines.append(f"\n=== {name} - K-Means ===")
    h2 = (
        f"{'Model':<48} | {'Size':>6} | {'Year':>4} | {'ms/sent':>7} | "
        f"{'k':>2} | {'ARI':>6} | {'V-meas':>6}"
    )
    lines.extend([h2, "-" * len(h2)])
    for m in MODELS:
        r = km_results[m]
        lines.append(
            f"{m:<48} | {r['size']:>6} | {r['year']:>4} | {r['ms_per_sent']:>7.2f} | "
            f"{r['k']:>2} | {r['ari']:>6.4f} | {r['v_measure']:>6.4f}"
        )
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["massive", "sib200"], required=True)
    args = parser.parse_args()

    api = HfApi()
    loaded_models = {}
    for model_name in MODELS:
        kwargs = {"trust_remote_code": True}
        if "jina-embeddings-v5" in model_name:
            kwargs["model_kwargs"] = {"default_task": "clustering"}
        emb = SentenceTransformer(model_name, **kwargs)
        size_str, year = get_hf_metadata(api, model_name, emb)
        loaded_models[model_name] = (emb, size_str, year)

    if args.dataset == "massive":
        texts, labels = load_massive_es()
        title = "MASSIVE (es-ES, validation, 1000 docs)"
        out_json = "results_massive.json"
        out_txt_path = "results_massive.txt"
    else:
        texts, labels = load_sib200_es()
        title = "SIB200 (spa_Latn)"
        out_json = "results_sib200.json"
        out_txt_path = "results_sib200.txt"

    bt, km = evaluate_dataset(title, texts, labels, loaded_models, api)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"bertopic": bt, "kmeans": km}, f, indent=2)

    out_txt = format_tables(title, bt, km) + "\n"
    print("\n" + out_txt)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(out_txt)


if __name__ == "__main__":
    main()
