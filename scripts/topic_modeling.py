# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "bertopic",
#   "pandas",
#   "pyarrow",
#   "sentence-transformers",
# ]
# ///

import argparse
import json
import os
import re

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

MODEL_NAME = "hotchpotch/bekko-embedding-v1-a25m"

_RT_RE = re.compile(r"\bRT\s+@\w+:?\s*", flags=re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")


def clean_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _RT_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def load_input(path: str, text_col: str | None, id_col: str | None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f if line.strip()]
        return pd.DataFrame({"id": range(len(lines)), "text": lines})
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {ext}. Use .txt, .csv, or .parquet.")

    if not text_col or text_col not in df.columns:
        raise ValueError(f"--text-col '{text_col}' not found in columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["id"] = df[id_col] if (id_col and id_col in df.columns) else df.index
    out["text"] = df[text_col].astype(str)
    return out


def cmd_run(args):
    df = load_input(args.input, args.text_col, args.id_col)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    raw_texts = df["text"].tolist()
    cleaned_texts = [clean_tweet(t) for t in raw_texts]
    min_topic_size = max(5, min(50, int(0.015 * len(cleaned_texts))))

    emb_model = SentenceTransformer(MODEL_NAME)
    topic_model = BERTopic(embedding_model=emb_model, min_topic_size=min_topic_size)
    topics, probs = topic_model.fit_transform(cleaned_texts)

    os.makedirs(args.out_dir, exist_ok=True)
    clusters_path = os.path.join(args.out_dir, "clusters.parquet")
    clusters_df = pd.DataFrame({
        "id": df["id"].values,
        "topic": topics,
        "topic_prob": [float(p) if p is not None else 0.0 for p in probs],
    })
    clusters_df.to_parquet(clusters_path, index=False)

    info_df = topic_model.get_topic_info()
    total = len(df)
    summary = []
    for _, row in info_df.iterrows():
        tid = int(row["Topic"])
        count = int(row["Count"])
        words = [w for w, _ in (topic_model.get_topic(tid) or [])[:10]]
        rep_docs = topic_model.get_representative_docs(tid) or []
        summary.append({
            "topic_id": tid,
            "count": count,
            "pct": round((count / total) * 100.0, 2),
            "keywords": words,
            "representative_docs": rep_docs[:3],
        })

    summary_path = os.path.join(args.out_dir, "topics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Processed {total} docs (min_topic_size={min_topic_size}).")
    print(f"Saved: {clusters_path}")
    print(f"Saved: {summary_path}")


def cmd_inspect(args):
    clusters_df = pd.read_parquet(args.clusters)
    subset = clusters_df[clusters_df["topic"] == args.topic]
    if subset.empty:
        print(f"No documents found for topic {args.topic}.")
        return

    sample_df = subset.sample(n=min(args.n, len(subset)), random_state=args.seed)
    if args.input:
        orig_df = load_input(args.input, args.text_col, args.id_col)
        merged = sample_df.merge(orig_df, on="id", how="left")
        for _, row in merged.iterrows():
            print(f"[{row['id']}] (prob={row['topic_prob']:.3f}) {row['text']}")
    else:
        for _, row in sample_df.iterrows():
            print(f"id={row['id']} | topic={row['topic']} | prob={row['topic_prob']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Exploratory BERTopic clustering tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run BERTopic clustering")
    p_run.add_argument("--input", required=True, help="Path to .txt, .csv, or .parquet")
    p_run.add_argument("--text-col", default="text", help="Text column name (csv/parquet)")
    p_run.add_argument("--id-col", default=None, help="ID column name (optional)")
    p_run.add_argument("--sample", type=int, default=None, help="Sample size N")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.add_argument("--out-dir", default="./topic_run", help="Output directory")
    p_run.set_defaults(func=cmd_run)

    p_ins = sub.add_parser("inspect", help="Inspect samples from a topic")
    p_ins.add_argument("--clusters", required=True, help="Path to clusters.parquet")
    p_ins.add_argument("--topic", type=int, required=True, help="Topic ID to inspect")
    p_ins.add_argument("--n", type=int, default=10, help="Number of samples")
    p_ins.add_argument("--seed", type=int, default=42, help="Random seed")
    p_ins.add_argument("--input", default=None, help="Original input file to join text")
    p_ins.add_argument("--text-col", default="text", help="Text column name")
    p_ins.add_argument("--id-col", default=None, help="ID column name")
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
