"""Repeatable, multi-seed evaluation of CineMark recommendation baselines.

The source feedback has no event timestamps, so this script evaluates
held-out positive-item recommendation, not chronological Markov prediction.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RATINGS_PATH = DATA_DIR / "ratings.json"
MOVIES_PATH = DATA_DIR / "movies.csv"
TOP_K = 5
DEFAULT_SEEDS = (7, 21, 42, 73, 101)
MODEL_NAMES = (
    "Popularity",
    "Content TF-IDF + scaled metadata",
    "Item-item collaborative",
    "Hybrid content + collaborative",
)


def parse_listish(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return [text]


def load_and_clean_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if not RATINGS_PATH.is_file() or not MOVIES_PATH.is_file():
        raise FileNotFoundError(
            "Expected data/ratings.json and data/movies.csv beside run_benchmark.py."
        )

    movies_raw = pd.read_csv(MOVIES_PATH)
    required = {"movie_id", "name"}
    missing = required.difference(movies_raw.columns)
    if missing:
        raise ValueError(f"movies.csv is missing required columns: {sorted(missing)}")

    movies = movies_raw.copy()
    movies["movie_id"] = movies["movie_id"].fillna("").astype(str).str.strip()
    movies = movies[movies["movie_id"].str.startswith("tt")]
    movie_rows_before_dedup = len(movies)
    movies = movies.drop_duplicates("movie_id", keep="first").reset_index(drop=True)

    rating_rows: list[tuple[str, str, int]] = []
    bad_records = 0
    invalid_labels = 0
    non_movie_keys = 0
    for line_number, line in enumerate(
        RATINGS_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on ratings.json line {line_number}.") from exc
        user_id = str(record.get("_id", "")).strip()
        rated = record.get("rated", {})
        if not user_id or not isinstance(rated, dict):
            bad_records += 1
            continue
        for raw_movie_id, raw_value in rated.items():
            movie_id = str(raw_movie_id).strip()
            if not movie_id.startswith("tt"):
                non_movie_keys += 1
                continue
            value = raw_value[0] if isinstance(raw_value, list) and raw_value else raw_value
            try:
                label = int(value)
            except (TypeError, ValueError):
                invalid_labels += 1
                continue
            if label not in {-1, 0, 1}:
                invalid_labels += 1
                continue
            rating_rows.append((user_id, movie_id, label))

    feedback = pd.DataFrame(rating_rows, columns=["user_id", "movie_id", "label"])
    duplicate_pairs = int(feedback.duplicated(["user_id", "movie_id"]).sum())
    feedback = feedback.drop_duplicates(["user_id", "movie_id"], keep="last")
    known_movie_ids = set(movies["movie_id"])
    unmatched_feedback = int((~feedback["movie_id"].isin(known_movie_ids)).sum())
    feedback = feedback[feedback["movie_id"].isin(known_movie_ids)].reset_index(drop=True)
    positive = feedback[feedback["label"] == 1].copy()
    if positive.empty:
        raise ValueError("No positive movie feedback remains after cleaning.")

    summary = {
        "movie rows before deduplication": movie_rows_before_dedup,
        "unique movies": int(movies["movie_id"].nunique()),
        "feedback pairs after cleaning": len(feedback),
        "unique feedback users": int(feedback["user_id"].nunique()),
        "positive feedback pairs": int((feedback["label"] == 1).sum()),
        "neutral feedback pairs": int((feedback["label"] == 0).sum()),
        "negative feedback pairs": int((feedback["label"] == -1).sum()),
        "duplicate user-movie pairs removed": duplicate_pairs,
        "unmatched feedback rows removed": unmatched_feedback,
        "non-movie keys ignored": non_movie_keys,
        "invalid labels ignored": invalid_labels,
        "invalid feedback records": bad_records,
    }
    return movies, feedback, summary


def make_train_test_split(
    feedback: pd.DataFrame, seed: int
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    rng = np.random.RandomState(seed)
    train_seen: dict[str, set[str]] = {}
    train_positive: dict[str, set[str]] = {}
    test_target: dict[str, str] = {}
    positive = feedback[feedback["label"] == 1]
    all_by_user = feedback.groupby("user_id")["movie_id"].apply(set).to_dict()
    for user_id, group in positive.groupby("user_id", sort=True):
        positive_ids = group["movie_id"].tolist()
        if len(positive_ids) < 2:
            continue
        holdout = int(rng.randint(0, len(positive_ids)))
        target = positive_ids[holdout]
        test_target[user_id] = target
        train_seen[user_id] = all_by_user[user_id] - {target}
        train_positive[user_id] = set(positive_ids) - {target}
    if not test_target:
        raise ValueError("Need users with at least two positive movies to evaluate.")
    return train_seen, train_positive, test_target


def build_content_features(
    movies: pd.DataFrame,
) -> tuple[list[str], csr_matrix, np.ndarray]:
    text_columns = ("name", "description", "genre", "language", "cast", "director", "writer")
    for column in text_columns:
        if column not in movies:
            movies[column] = ""
    text_parts = []
    for row in movies[list(text_columns)].itertuples(index=False, name=None):
        cleaned = []
        for column, value in zip(text_columns, row):
            if column in {"genre", "language", "cast", "director", "writer"}:
                cleaned.extend(parse_listish(value))
            elif pd.notna(value):
                cleaned.append(str(value))
        text_parts.append(" ".join(cleaned).lower())
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=50000, ngram_range=(1, 2), min_df=1
    )
    text_matrix = vectorizer.fit_transform(text_parts).tocsr()

    numeric = pd.DataFrame(index=movies.index)
    numeric["catalogue_rating"] = pd.to_numeric(movies.get("rating"), errors="coerce")
    if "released" in movies:
        numeric["release_year"] = pd.to_datetime(
            movies["released"], errors="coerce", utc=True
        ).dt.year
    else:
        numeric["release_year"] = np.nan
    imputed = SimpleImputer(strategy="median", add_indicator=True).fit_transform(numeric)
    scaled_numeric = StandardScaler().fit_transform(imputed)
    scaled_numeric = normalize(scaled_numeric, norm="l2")
    return movies["movie_id"].tolist(), text_matrix, scaled_numeric


def _top_unseen(
    scores: np.ndarray, movie_ids: list[str], seen: set[str], k: int
) -> list[int]:
    order = np.lexsort((np.asarray(movie_ids), -scores))
    result: list[int] = []
    for index in order:
        if movie_ids[index] not in seen:
            result.append(int(index))
            if len(result) == k:
                break
    return result


def evaluate_seed(
    movies: pd.DataFrame,
    feedback: pd.DataFrame,
    movie_ids: list[str],
    text_matrix: csr_matrix,
    scaled_numeric: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_seen, train_positive, targets = make_train_test_split(feedback, seed)
    movie_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
    user_ids = sorted(train_seen)
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    rows = [
        (user_to_index[user_id], movie_to_index[movie_id])
        for user_id, liked in train_positive.items()
        for movie_id in liked
    ]
    interaction_matrix = csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            ([row[0] for row in rows], [row[1] for row in rows]),
        ),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )
    item_similarity = cosine_similarity(
        interaction_matrix.T, dense_output=False
    ).tocsr()
    item_similarity.setdiag(0)
    item_similarity.eliminate_zeros()
    popularity = np.asarray(interaction_matrix.sum(axis=0)).ravel()

    metric_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    title_by_id = movies.set_index("movie_id")["name"].fillna("").to_dict()
    detail_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_NAMES}

    for user_number, user_id in enumerate(sorted(targets)):
        seen = train_seen[user_id]
        liked_indices = [
            movie_to_index[mid]
            for mid in train_positive[user_id]
            if mid in movie_to_index
        ]
        content_profile = text_matrix[liked_indices].mean(axis=0)
        content_scores = np.asarray(text_matrix @ content_profile.T).ravel()
        content_norm = float(np.linalg.norm(content_profile))
        if content_norm:
            content_scores /= content_norm

        numeric_profile = scaled_numeric[liked_indices].mean(axis=0)
        numeric_norm = float(np.linalg.norm(numeric_profile))
        if numeric_norm:
            numeric_profile /= numeric_norm
        numeric_scores = (scaled_numeric @ numeric_profile + 1.0) / 2.0
        content_scores = 0.85 * content_scores + 0.15 * numeric_scores

        collaborative_scores = np.zeros(len(movie_ids), dtype=np.float64)
        user_row = interaction_matrix.getrow(user_to_index[user_id])
        if user_row.nnz:
            collaborative_scores = np.asarray(
                user_row.dot(item_similarity).toarray()
            ).ravel()

        score_by_model = {
            "Popularity": popularity.astype(np.float64),
            "Content TF-IDF + scaled metadata": content_scores,
            "Item-item collaborative": collaborative_scores,
        }
        content_scaled = _scale_scores(content_scores, seen, movie_ids)
        collaborative_scaled = _scale_scores(
            collaborative_scores, seen, movie_ids
        )
        score_by_model["Hybrid content + collaborative"] = (
            0.5 * content_scaled + 0.5 * collaborative_scaled
        )

        target = targets[user_id]
        for model_name in MODEL_NAMES:
            top_indices = _top_unseen(
                score_by_model[model_name], movie_ids, seen, TOP_K
            )
            target_rank = next(
                (rank for rank, idx in enumerate(top_indices, start=1) if movie_ids[idx] == target),
                None,
            )
            detail_rows[model_name].append(
                {
                    "hit_rate_at_5": int(target_rank is not None),
                    "recall_at_5": int(target_rank is not None),
                    "mrr_at_5": 1.0 / target_rank if target_rank is not None else 0.0,
                    "ndcg_at_5": (
                        1.0 / np.log2(target_rank + 1)
                        if target_rank is not None
                        else 0.0
                    ),
                }
            )
            if seed == 42 and user_number < 3:
                masked_user = f"test_user_{user_number + 1:03d}"
                for rank, index in enumerate(top_indices, start=1):
                    preview_rows.append(
                        {
                            "seed": seed,
                            "masked_test_user": masked_user,
                            "model": model_name,
                            "rank": rank,
                            "recommended_movie": title_by_id.get(movie_ids[index], ""),
                            "is_held_out_target": movie_ids[index] == target,
                        }
                    )

    for model_name, details in detail_rows.items():
        detail = pd.DataFrame(details)
        metric_rows.append(
            {
                "seed": seed,
                "model": model_name,
                "hit_rate_at_5": detail["hit_rate_at_5"].mean(),
                "recall_at_5": detail["recall_at_5"].mean(),
                "mrr_at_5": detail["mrr_at_5"].mean(),
                "ndcg_at_5": detail["ndcg_at_5"].mean(),
                "test_users": len(detail),
            }
        )
    return metric_rows, preview_rows


def _scale_scores(scores: np.ndarray, seen: set[str], movie_ids: list[str]) -> np.ndarray:
    available = np.fromiter(
        (movie_id not in seen for movie_id in movie_ids), dtype=bool, count=len(movie_ids)
    )
    values = scores[available]
    low = float(values.min()) if values.size else 0.0
    high = float(values.max()) if values.size else 0.0
    scaled = np.zeros_like(scores, dtype=np.float64)
    if high > low:
        scaled[available] = (values - low) / (high - low)
    return scaled


def run_synthetic_markov(seed: int, n_users: int = 500, n_states: int = 40,
                         events_per_user: int = 30, k: int = TOP_K) -> dict[str, Any]:
    """Train and test a first-order chain on generated sequences, never user data."""
    rng = np.random.RandomState(seed)
    transition = np.full((n_states, n_states), 0.01, dtype=np.float64)
    for state in range(n_states):
        destinations = rng.choice(
            [other for other in range(n_states) if other != state],
            size=min(4, n_states - 1),
            replace=False,
        )
        transition[state, destinations] += rng.dirichlet(
            np.full(len(destinations), 2.0)
        )
    transition /= transition.sum(axis=1, keepdims=True)
    start_probability = rng.dirichlet(np.ones(n_states))
    sequences: list[list[int]] = []
    for _ in range(n_users):
        sequence = [int(rng.choice(n_states, p=start_probability))]
        for _ in range(events_per_user - 1):
            sequence.append(int(rng.choice(n_states, p=transition[sequence[-1]])))
        sequences.append(sequence)

    split_at = int(events_per_user * 0.8)
    transition_counts = np.zeros_like(transition)
    popularity = np.zeros(n_states, dtype=np.float64)
    for sequence in sequences:
        train_sequence = sequence[:split_at]
        for current, following in zip(train_sequence, train_sequence[1:]):
            transition_counts[current, following] += 1
        for state in train_sequence:
            popularity[state] += 1

    smoothing = 0.1
    learned_transition = (transition_counts + smoothing) / (
        transition_counts.sum(axis=1, keepdims=True) + smoothing * n_states
    )
    hit_counts = {"Markov chain": 0, "Popularity": 0}
    reciprocal_rank = {"Markov chain": 0.0, "Popularity": 0.0}
    ndcg = {"Markov chain": 0.0, "Popularity": 0.0}
    test_events = 0
    state_ids = [f"state_{i:03d}" for i in range(n_states)]
    for sequence in sequences:
        for position in range(split_at, events_per_user):
            current, target = sequence[position - 1], sequence[position]
            for model, scores in (
                ("Markov chain", learned_transition[current]),
                ("Popularity", popularity),
            ):
                order = np.lexsort((np.asarray(state_ids), -scores))
                rank = next(
                    (i + 1 for i, state in enumerate(order) if state == target),
                    None,
                )
                if rank is not None and rank <= k:
                    hit_counts[model] += 1
                    reciprocal_rank[model] += 1.0 / rank
                    ndcg[model] += 1.0 / np.log2(rank + 1)
            test_events += 1
    return {
        "seed": seed,
        "synthetic_users": n_users,
        "states": n_states,
        "train_events_per_user": split_at,
        "test_next_events": test_events,
        "markov_hit_rate_at_5": hit_counts["Markov chain"] / test_events,
        "markov_mrr_at_5": reciprocal_rank["Markov chain"] / test_events,
        "markov_ndcg_at_5": ndcg["Markov chain"] / test_events,
        "popularity_hit_rate_at_5": hit_counts["Popularity"] / test_events,
        "popularity_mrr_at_5": reciprocal_rank["Popularity"] / test_events,
        "popularity_ndcg_at_5": ndcg["Popularity"] / test_events,
        "mean_transition_matrix_l1_error": float(
            np.abs(learned_transition - transition).sum(axis=1).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds for independent per-user holdout splits (default: 7 21 42 73 101).",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide each random seed only once.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    movies, feedback, quality = load_and_clean_data()
    movie_ids, text_matrix, scaled_numeric = build_content_features(movies)
    all_metrics: list[dict[str, Any]] = []
    all_previews: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"Evaluating seed {seed}...")
        seed_metrics, previews = evaluate_seed(
            movies, feedback, movie_ids, text_matrix, scaled_numeric, seed
        )
        all_metrics.extend(seed_metrics)
        all_previews.extend(previews)

    seed_results = pd.DataFrame(all_metrics)
    metric_columns = (
        "hit_rate_at_5",
        "recall_at_5",
        "mrr_at_5",
        "ndcg_at_5",
    )
    summary_rows = []
    for model_name, group in seed_results.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model_name,
            "seeds_evaluated": len(group),
            "test_users_per_seed": int(group["test_users"].min()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["ndcg_at_5_mean", "hit_rate_at_5_mean", "mrr_at_5_mean"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    summary.insert(0, "rank_by_mean_ndcg_at_5", np.arange(1, len(summary) + 1))
    best_model = str(summary.iloc[0]["model"])

    pd.DataFrame(list(quality.items()), columns=["check", "value"]).to_csv(
        OUTPUT_DIR / "benchmark_data_quality.csv", index=False
    )
    seed_results.to_csv(OUTPUT_DIR / "seed_model_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "model_seed_summary.csv", index=False)
    pd.DataFrame(all_previews).to_csv(
        OUTPUT_DIR / "recommendation_preview.csv", index=False
    )
    synthetic_markov = pd.DataFrame(
        [run_synthetic_markov(seed) for seed in args.seeds]
    )
    synthetic_markov.to_csv(
        OUTPUT_DIR / "synthetic_markov_seed_metrics.csv", index=False
    )
    markov_summary = synthetic_markov.drop(
        columns=["seed", "synthetic_users", "states", "train_events_per_user",
                 "test_next_events"]
    ).agg(["mean", "std"]).T.reset_index(names="metric")
    markov_summary.to_csv(
        OUTPUT_DIR / "synthetic_markov_summary.csv", index=False
    )

    plot = summary.set_index("model")[
        ["hit_rate_at_5_mean", "mrr_at_5_mean", "ndcg_at_5_mean"]
    ].plot(kind="bar", figsize=(11, 6))
    plot.set_title("Recommendation quality across seeded per-user holdouts")
    plot.set_ylabel("Mean metric across random seeds")
    plot.set_xlabel("")
    plot.legend(
        ["Hit Rate@5", "MRR@5", "NDCG@5"],
        loc="upper right",
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multi_seed_model_comparison.png", dpi=180)
    plt.close()

    markov_plot = synthetic_markov[
        ["markov_hit_rate_at_5", "popularity_hit_rate_at_5"]
    ].agg(["mean", "std"]).T
    markov_plot.columns = ["mean", "std"]
    ax = markov_plot["mean"].plot(
        kind="bar",
        yerr=markov_plot["std"].fillna(0),
        capsize=4,
        figsize=(7, 5),
        color=["#7c3aed", "#f97316"],
    )
    ax.set_title("Synthetic sequences only: next-event Hit Rate@5")
    ax.set_ylabel("Mean across random seeds")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "synthetic_markov_seed_comparison.png", dpi=180)
    plt.close()

    print(f"\nPositive-feedback train/test seeds: {args.seeds}")
    print(f"Evaluation users per seed: {int(seed_results['test_users'].min()):,}")
    print(f"Best model by mean NDCG@5 (tie-break: Hit Rate@5, MRR@5): {best_model}")
    display_columns = [
        "rank_by_mean_ndcg_at_5",
        "model",
        "hit_rate_at_5_mean",
        "hit_rate_at_5_std",
        "mrr_at_5_mean",
        "ndcg_at_5_mean",
        "ndcg_at_5_std",
    ]
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved benchmark tables and chart to: {OUTPUT_DIR}")
    print(
        "Separate synthetic Markov next-event results "
        "(not measured on CineMark users):"
    )
    print(
        synthetic_markov[
            [
                "seed",
                "markov_hit_rate_at_5",
                "popularity_hit_rate_at_5",
                "mean_transition_matrix_l1_error",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
