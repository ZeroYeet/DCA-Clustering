from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from benchmark_data import load_csv_file
from dca import ip_dca


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


def resolve_cluster_count(
    true_labels: Any,
    requested_count: int | None,
) -> int:
    if requested_count is not None:
        return requested_count
    if true_labels is None:
        raise ValueError("n_clusters is required when the dataset has no target column")
    return len(set(true_labels.tolist()))


def _sample_indices_preserving_clusters(
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
) -> np.ndarray | None:
    _, cluster_ids = np.unique(labels, return_inverse=True)
    cluster_count = int(cluster_ids.max()) + 1
    if cluster_count > sample_size:
        return None

    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(len(labels), size=sample_size, replace=False)
    sampled_cluster_ids = cluster_ids[sample_indices]
    sampled_counts = np.bincount(sampled_cluster_ids, minlength=cluster_count)

    for cluster_id in np.flatnonzero(sampled_counts == 0):
        replaceable_positions = np.flatnonzero(
            sampled_counts[sampled_cluster_ids] > 1
        )
        replace_position = int(rng.choice(replaceable_positions))
        previous_cluster_id = sampled_cluster_ids[replace_position]
        sampled_cluster_ids[replace_position] = cluster_id
        sample_indices[replace_position] = rng.choice(
            np.flatnonzero(cluster_ids == cluster_id)
        )
        sampled_counts[cluster_id] += 1
        sampled_counts[previous_cluster_id] -= 1

    return sample_indices


def evaluate_dataset(
    dataset_path: Path,
    *,
    target_column: str | None = None,
    has_header: bool | None = None,
    include_target: bool = True,
    n_clusters: int | None = None,
    t: float = 10.0,
    n_init: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: int = 42,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run IP-DCA on one CSV dataset from the local data directory."""
    features, true_labels = load_csv_file(
        dataset_path,
        target_column=target_column,
        has_header=has_header,
        include_target=include_target,
    )
    cluster_count = resolve_cluster_count(true_labels, n_clusters)

    start = time.perf_counter()
    result = ip_dca(
        features,
        n_clusters=cluster_count,
        t=t,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=n_init,
        progress_callback=progress_callback,
    )
    runtime_seconds = time.perf_counter() - start

    predicted_labels = result["labels"]
    predicted_cluster_count = len(set(predicted_labels.tolist()))
    silhouette = None
    davies_bouldin_index = None
    ari = None
    sample_size = min(2000, len(predicted_labels))
    if sample_size < len(predicted_labels):
        sample_indices = _sample_indices_preserving_clusters(
            predicted_labels,
            sample_size,
            random_state,
        )
    else:
        sample_indices = np.arange(len(predicted_labels))

    if sample_indices is not None:
        metric_features = features[sample_indices]
        metric_labels = predicted_labels[sample_indices]
        metric_cluster_count = len(set(metric_labels.tolist()))
        if 1 < metric_cluster_count < len(metric_labels):
            silhouette = float(
                silhouette_score(metric_features, metric_labels) * 100
            )
            davies_bouldin_index = float(
                davies_bouldin_score(metric_features, metric_labels)
            )
    if true_labels is not None:
        ari = float(adjusted_rand_score(true_labels, predicted_labels))

    return {
        "dataset": dataset_path.name,
        "samples": features.shape[0],
        "features": features.shape[1],
        "clusters": predicted_cluster_count,
        "silhouette": silhouette,
        "davies_bouldin_index": davies_bouldin_index,
        "ari": ari,
        "runtime_seconds": runtime_seconds,
        "iterations": result["iterations"],
    }


def print_results(results: list[dict[str, Any]]) -> None:
    columns = (
        ("dataset", "Dataset"),
        ("samples", "Samples"),
        ("features", "Features"),
        ("clusters", "Clusters"),
        ("silhouette", "Silhouette"),
        ("runtime_seconds", "Runtime (s)"),
    )
    widths = {
        key: max(len(title), *(len(str(result[key])) for result in results))
        for key, title in columns
    }
    print("  ".join(title.ljust(widths[key]) for key, title in columns))
    for result in results:
        values = []
        for key, _ in columns:
            value = result[key]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).ljust(widths[key]))
        print("  ".join(values))


def save_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def save_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    metric_names = ("silhouette",)
    metrics = [
        name
        for name in metric_names
        if any(result[name] is not None for result in results)
    ]
    if not metrics:
        raise ValueError("No metrics are available to plot")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(8, len(results) * 1.5), 5))
    width = 0.8 / len(metrics)
    positions = list(range(len(results)))
    for metric_index, metric in enumerate(metrics):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        values = [
            result[metric] if result[metric] is not None else 0
            for result in results
        ]
        axis.bar(
            [position + offset for position in positions],
            values,
            width,
            label=metric.upper(),
        )

    axis.set_xticks(positions, [result["dataset"] for result in results])
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def show_progress(
    completed: int,
    total: int,
    dataset_name: str,
    start_time: float,
) -> None:
    elapsed = time.perf_counter() - start_time
    eta_text = (
        format_duration(elapsed / completed * (total - completed))
        if completed
        else "--:--"
    )
    fraction = completed / total
    bar_width = 30
    filled = int(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    message = (
        f"\r[{bar}] {fraction:6.1%} | {dataset_name} | "
        f"ETA {eta_text}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()
    if completed == total:
        sys.stdout.write("\n")


def main() -> None:
    dataset_paths = sorted(DATA_DIR.glob("*.csv"))
    if not dataset_paths:
        raise FileNotFoundError(f"No CSV datasets found in {DATA_DIR}")

    dataset_names = [path.stem for path in dataset_paths]
    parser = argparse.ArgumentParser(
        description="Evaluate IP-DCA on CSV datasets in the local data directory."
    )
    parser.add_argument(
        "--dataset",
        choices=(*dataset_names, "all"),
        default="all",
        help="Dataset filename without .csv, or all datasets (default: all).",
    )
    parser.add_argument(
        "--target-column",
        help="Target column name or zero-based index (default: last column).",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Treat every CSV column as a feature and evaluate with Silhouette only.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Cluster count; required with --no-target, otherwise inferred from labels.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the CSV as headerless.",
    )
    parser.add_argument("--t", type=float, default=10.0, help="DCA penalty parameter.")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "dca_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=RESULTS_DIR / "dca_results.png",
        help="Chart output path (PNG, SVG, or PDF).",
    )
    args = parser.parse_args()

    if args.no_target and args.target_column is not None:
        parser.error("--target-column cannot be used with --no-target")
    if args.no_target and args.n_clusters is None:
        parser.error("--n-clusters is required with --no-target")

    selected_paths = (
        dataset_paths
        if args.dataset == "all"
        else [path for path in dataset_paths if path.stem == args.dataset]
    )
    results = []
    total_runs = len(selected_paths) * args.n_init
    progress_started = time.perf_counter()
    show_progress(0, total_runs, selected_paths[0].name, progress_started)
    for dataset_index, path in enumerate(selected_paths):
        def update_progress(completed: int, _: int) -> None:
            overall_completed = dataset_index * args.n_init + completed
            show_progress(
                overall_completed,
                total_runs,
                path.name,
                progress_started,
            )

        results.append(
            evaluate_dataset(
                path,
                target_column=args.target_column,
                has_header=False if args.no_header else None,
                include_target=not args.no_target,
                n_clusters=args.n_clusters,
                t=args.t,
                n_init=args.n_init,
                max_iter=args.max_iter,
                tol=args.tol,
                random_state=args.random_state,
                progress_callback=update_progress,
            )
        )

    print_results(results)
    save_csv(results, args.output)
    print(f"\nCSV exported to: {args.output}")
    save_plot(results, args.plot, title="IP-DCA evaluation metrics")
    print(f"Chart exported to: {args.plot}")


if __name__ == "__main__":
    main()
