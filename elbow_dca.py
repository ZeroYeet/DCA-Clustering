from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_data import load_csv_file
from dca import ip_dca


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


def estimate_elbow_k(cluster_counts: list[int], objectives: list[float]) -> int | None:
    """Estimate the elbow as the point farthest from the endpoint chord."""
    if len(cluster_counts) != len(objectives):
        raise ValueError("cluster_counts and objectives must have the same length")
    if len(cluster_counts) < 3:
        raise ValueError("At least three K values are required to estimate an elbow")

    x = np.asarray(cluster_counts, dtype=float)
    y = np.asarray(objectives, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Objective values must be finite")
    objective_range = float(np.ptp(y))
    if objective_range == 0:
        return None

    x = (x - x[0]) / (x[-1] - x[0])
    y = (y - y.min()) / objective_range
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    chord = end - start
    distances = np.abs(
        chord[0] * (start[1] - y) - (start[0] - x) * chord[1]
    ) / np.linalg.norm(chord)
    distances[[0, -1]] = -1
    return int(cluster_counts[int(np.argmax(distances))])


def run_elbow_search(
    dataset_path: Path,
    *,
    min_k: int = 2,
    max_k: int = 10,
    target_column: str | None = None,
    has_header: bool | None = None,
    include_target: bool = True,
    sample_size: int = 5000,
    n_init: int = 3,
    t: float = 10.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], int | None]:
    features, _ = load_csv_file(
        dataset_path,
        target_column=target_column,
        has_header=has_header,
        include_target=include_target,
    )
    if min_k < 2:
        raise ValueError("min_k must be at least 2")
    if max_k < min_k + 2:
        raise ValueError("max_k must be at least min_k + 2 to estimate an elbow")
    if max_k >= features.shape[0]:
        raise ValueError("max_k must be smaller than the number of samples")
    if sample_size < max_k:
        raise ValueError("sample_size must be at least max_k")
    if n_init <= 0:
        raise ValueError("n_init must be positive")

    if features.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(
            features.shape[0], size=sample_size, replace=False
        )
        features = features[sample_indices]

    results = []
    total_runs = (max_k - min_k + 1) * n_init
    completed_runs = 0
    search_started = time.perf_counter()
    for cluster_count in range(min_k, max_k + 1):
        start = time.perf_counter()

        def update_progress(completed: int, _: int) -> None:
            nonlocal completed_runs
            completed_runs += 1
            elapsed = time.perf_counter() - search_started
            eta_seconds = elapsed / completed_runs * (total_runs - completed_runs)
            print(
                f"\rK={cluster_count}: initialization {completed}/{n_init} "
                f"| total {completed_runs}/{total_runs} "
                f"| ETA {int(eta_seconds)}s",
                end="",
                flush=True,
            )

        result = ip_dca(
            features,
            n_clusters=cluster_count,
            t=t,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state + cluster_count,
            n_init=n_init,
            progress_callback=update_progress,
        )
        results.append(
            {
                "k": cluster_count,
                "mssc": result["mssc"],
                "penalized_objective": result["penalized_objective"],
                "clusters_found": len(np.unique(result["labels"])),
                "iterations": result["iterations"],
                "runtime_seconds": time.perf_counter() - start,
            }
        )

    print()
    elbow_k = estimate_elbow_k(
        [result["k"] for result in results],
        [result["mssc"] for result in results],
    )
    for result in results:
        result["estimated_elbow_k"] = elbow_k
    return results, elbow_k


def save_results_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def save_elbow_plot(
    results: list[dict[str, Any]],
    elbow_k: int | None,
    output_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    cluster_counts = [result["k"] for result in results]
    objectives = [result["mssc"] for result in results]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(cluster_counts, objectives, marker="o")
    if elbow_k is not None:
        elbow_objective = objectives[cluster_counts.index(elbow_k)]
        axis.scatter(
            [elbow_k],
            [elbow_objective],
            color="red",
            zorder=3,
            label=f"Estimated elbow: K={elbow_k}",
        )
        axis.axvline(elbow_k, color="red", linestyle="--", alpha=0.5)
        axis.legend()
    axis.set_xticks(cluster_counts)
    axis.set_xlabel("Number of clusters (K)")
    axis.set_ylabel("MSSC objective")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    dataset_paths = sorted(DATA_DIR.glob("*.csv"))
    if not dataset_paths:
        raise FileNotFoundError(f"No CSV datasets found in {DATA_DIR}")

    parser = argparse.ArgumentParser(
        description="Estimate the DCA cluster count using the elbow method."
    )
    parser.add_argument(
        "--dataset",
        choices=[path.stem for path in dataset_paths],
        required=True,
        help="Dataset filename without the .csv extension.",
    )
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--target-column",
        help="Target column to exclude from clustering; defaults to the last column.",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Treat every CSV column as a feature.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the CSV as having no header row.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Maximum rows used for all K values (default: 5000).",
    )
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--t", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    if args.no_target and args.target_column is not None:
        parser.error("--target-column cannot be used with --no-target")
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if args.n_init <= 0:
        parser.error("--n-init must be positive")
    if args.t <= 0:
        parser.error("--t must be positive")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if args.tol < 0:
        parser.error("--tol must be non-negative")

    dataset_path = next(path for path in dataset_paths if path.stem == args.dataset)
    results, elbow_k = run_elbow_search(
        dataset_path,
        min_k=args.min_k,
        max_k=args.max_k,
        target_column=args.target_column,
        has_header=False if args.no_header else None,
        include_target=not args.no_target,
        sample_size=args.sample_size,
        n_init=args.n_init,
        t=args.t,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.random_state,
    )

    output_path = args.output or RESULTS_DIR / f"{args.dataset}_elbow.csv"
    plot_path = args.plot or RESULTS_DIR / f"{args.dataset}_elbow.png"
    save_results_csv(results, output_path)
    save_elbow_plot(
        results,
        elbow_k,
        plot_path,
        title=f"IP-DCA elbow analysis: {args.dataset}",
    )
    print(f"\nEstimated elbow K: {elbow_k if elbow_k is not None else 'not found'}")
    print(f"CSV exported to: {output_path}")
    print(f"Plot exported to: {plot_path}")


if __name__ == "__main__":
    main()
