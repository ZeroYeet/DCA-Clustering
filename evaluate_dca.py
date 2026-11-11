from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
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


def save_pca_plot(
    features: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray | None,
    output_path: Path,
    *,
    random_state: int,
    sample_limit: int = 5000,
) -> None:
    import matplotlib.pyplot as plt

    sample_size = min(sample_limit, len(predicted_labels))
    if true_labels is None:
        sample_labels = predicted_labels
    else:
        sample_labels = np.asarray(
            [
                f"{cluster}\0{label}"
                for cluster, label in zip(predicted_labels, true_labels)
            ]
        )

    sample_indices = _sample_indices_preserving_clusters(
        sample_labels,
        sample_size,
        random_state,
    )
    if sample_indices is None:
        sample_indices = np.random.default_rng(random_state).choice(
            len(predicted_labels),
            size=sample_size,
            replace=False,
        )

    sampled_features = features[sample_indices]
    component_count = min(2, sampled_features.shape[0], sampled_features.shape[1])
    pca = PCA(n_components=component_count, random_state=random_state)
    projected = pca.fit_transform(sampled_features)
    if component_count == 1:
        projected = np.column_stack((projected[:, 0], np.zeros(sample_size)))

    label_sets = [("DCA clusters", predicted_labels[sample_indices])]
    if true_labels is not None:
        label_sets.append(("True labels", true_labels[sample_indices]))

    figure, axes = plt.subplots(
        1,
        len(label_sets),
        figsize=(8 * len(label_sets), 6),
        squeeze=False,
    )
    explained_variance = pca.explained_variance_ratio_ * 100
    for axis, (label_name, labels) in zip(axes[0], label_sets):
        categories, category_codes = np.unique(labels, return_inverse=True)
        scatter = axis.scatter(
            projected[:, 0],
            projected[:, 1],
            c=category_codes,
            cmap="tab20",
            s=8,
            alpha=0.6,
            rasterized=True,
        )
        colorbar = figure.colorbar(
            scatter,
            ax=axis,
            ticks=np.arange(len(categories)),
        )
        colorbar.ax.set_yticklabels([str(category) for category in categories])
        axis.set_title(f"PCA colored by {label_name}")
        axis.set_xlabel(f"PC1 ({explained_variance[0]:.1f}% variance)")
        axis.set_ylabel(
            f"PC2 ({explained_variance[1]:.1f}% variance)"
            if len(explained_variance) > 1
            else "PC2 (0% variance)"
        )

    figure.suptitle(
        f"PCA visualization: {len(sample_indices):,} of "
        f"{len(predicted_labels):,} samples"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


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
    pca_plot_path: Path | None = None,
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
            silhouette = float(silhouette_score(metric_features, metric_labels))
    if true_labels is not None:
        ari = float(adjusted_rand_score(true_labels, predicted_labels))
    if pca_plot_path is not None:
        save_pca_plot(
            features,
            predicted_labels,
            true_labels,
            pca_plot_path,
            random_state=random_state,
        )

    return {
        "dataset": dataset_path.name,
        "samples": features.shape[0],
        "features": features.shape[1],
        "clusters": predicted_cluster_count,
        "silhouette": silhouette,
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
        ("ari", "ARI"),
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
        help=(
            "Treat every CSV column as a feature and evaluate using internal "
            "metrics only (Silhouette)."
        ),
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
        help=(
            "CSV output path (default: results/<dataset>_dca_results.csv "
            "for one dataset, results/dca_results.csv for all datasets)."
        ),
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
    default_output = (
        RESULTS_DIR / f"{selected_paths[0].stem}_dca_results.csv"
        if len(selected_paths) == 1
        else RESULTS_DIR / "dca_results.csv"
    )
    output_path = args.output or default_output
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
                pca_plot_path=RESULTS_DIR / f"{path.stem}_pca.png",
            )
        )

    print_results(results)
    save_csv(results, output_path)
    print(f"\nCSV exported to: {output_path}")


if __name__ == "__main__":
    main()
