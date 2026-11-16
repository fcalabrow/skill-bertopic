from collections import Counter
from datasets import concatenate_datasets, load_dataset
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score


MODELS = [
    "hotchpotch/bekko-embedding-v1-a8m",
    "codefuse-ai/F2LLM-v2-160M",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
]


def load_spanish_tweets():
    ds = load_dataset("cardiffnlp/tweet_topic_multilingual", "es")
    full_ds = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    texts = full_ds["text"]
    # Multilabel binary vectors -> list of active label indices per document
    multi_labels = [
        [idx for idx, val in enumerate(row) if val == 1]
        for row in full_ds["label"]
    ]
    label_sets = full_ds["label_name_flatten"]
    return texts, multi_labels, label_sets


def optimistic_single_labels(multi_labels, pred_topics):
    """
    For each predicted cluster, find the most frequent true label among its documents.
    Then for each document, pick the valid label that best matches its cluster's
    majority label (if any), treating all categories in the multilabel set as valid.
    """
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


def main():
    texts, multi_labels, label_sets = load_spanish_tweets()
    print(f"Loaded {len(texts)} Spanish tweets.\n")

    results = []
    for model_name in MODELS:
        print(f"=== Evaluating: {model_name} ===")
        embedding_model = SentenceTransformer(model_name)
        topic_model = BERTopic(embedding_model=embedding_model)
        topics, _ = topic_model.fit_transform(texts)

        n_topics = len(set(topics) - {-1})
        n_outliers = sum(1 for t in topics if t == -1)

        resolved_labels = optimistic_single_labels(multi_labels, topics)
        ari_multilabel = adjusted_rand_score(resolved_labels, topics)
        ari_labelset = adjusted_rand_score(label_sets, topics)

        results.append({
            "model": model_name,
            "topics": n_topics,
            "outliers": n_outliers,
            "ari_multilabel": ari_multilabel,
            "ari_labelset": ari_labelset,
        })
        print(
            f"Topics: {n_topics} | Outliers: {n_outliers}/{len(texts)} | "
            f"ARI (multilabel valid): {ari_multilabel:.4f} | "
            f"ARI (exact label-set): {ari_labelset:.4f}\n"
        )

    print("=== Summary ===")
    print(f"{'Model':<48} | {'Topics':>6} | {'Outliers':>8} | {'ARI (multi)':>11} | {'ARI (set)':>9}")
    print("-" * 92)
    for r in results:
        print(
            f"{r['model']:<48} | {r['topics']:>6} | {r['outliers']:>8} | "
            f"{r['ari_multilabel']:>11.4f} | {r['ari_labelset']:>9.4f}"
        )


if __name__ == "__main__":
    main()
