from collections.abc import Callable

import numpy as np


def _sq_dists(V, X):
    """D[i, k] = ||x_k - v_i||^2, kích thước (c, n)."""
    cluster_count, sample_count = V.shape[0], X.shape[0]
    feature_count = X.shape[1]
    if sample_count == 0:
        return np.empty((cluster_count, 0), dtype=float)

    origin = X[0]
    centered_V = V - origin
    centered_V_norms = np.einsum("cd,cd->c", centered_V, centered_V)
    distances = np.empty((cluster_count, sample_count), dtype=float)
    max_chunk_elements = 2_000_000
    chunk_size = max(
        1,
        min(sample_count, max_chunk_elements // max(feature_count, 1)),
    )
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        centered_X = X[start:end] - origin
        block_distances = centered_V @ centered_X.T
        block_distances *= -2.0
        block_distances += centered_V_norms[:, None]
        block_distances += np.einsum("nd,nd->n", centered_X, centered_X)[None, :]
        np.maximum(block_distances, 0.0, out=block_distances)
        distances[:, start:end] = block_distances
    return distances


def _proj_simplex(Y):
    """Chiếu từng cột của Y (c x n) lên đơn hình {y >= 0, sum y = 1}."""
    c, n = Y.shape
    S = -np.sort(-Y, axis=0)
    css = np.cumsum(S, axis=0) - 1.0
    idx = np.arange(1, c + 1)[:, None]
    k = ((S - css / idx) > 0).sum(axis=0)
    theta = css[k - 1, np.arange(n)] / k
    return np.maximum(Y - theta, 0.0)


def _proj_ball(Z, r):
    """Chiếu từng hàng của Z (c x d) lên hình cầu tâm 0, bán kính r."""
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z * np.where(norms <= r, 1.0, r / np.maximum(norms, 1e-300))


def mssc_objective(X, U, V):
    """F(U, V) = sum_k sum_i u_ik^2 ||x_k - v_i||^2."""
    return float((U ** 2 * _sq_dists(V, X)).sum())


def hard_mssc_objective(X, labels):
    """MSSC for fixed labels, using each cluster's arithmetic-mean center."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or labels.ndim != 1 or labels.shape[0] != X.shape[0]:
        raise ValueError("X must be 2D and labels must have one entry per sample")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample")

    _, cluster_ids = np.unique(labels, return_inverse=True)
    cluster_count = int(cluster_ids.max()) + 1
    counts = np.bincount(cluster_ids, minlength=cluster_count)
    centers = np.zeros((cluster_count, X.shape[1]), dtype=float)
    np.add.at(centers, cluster_ids, X)
    centers /= counts[:, None]
    return float(((X - centers[cluster_ids]) ** 2).sum())


def penalty(U):
    """p(U) = sum_i sum_k u_ik (1 - u_ik); bằng 0 khi U là ma trận 0/1."""
    return float((U * (1.0 - U)).sum())


def normalized_penalty(U):
    """Penalty divided by its maximum over column-simplex memberships."""
    cluster_count, sample_count = U.shape
    return penalty(U) / (sample_count * (1.0 - 1.0 / cluster_count))


def penalized_objective(X, U, V, t):
    """F1(U, V) = F(U, V) + t * p(U)   (công thức (20))."""
    return mssc_objective(X, U, V) + t * penalty(U)


def _rho_bound(X, r):
    """Cận rho theo (15): n[(alpha^2/n + 1) + sqrt((alpha^2/n + 1)^2 + 16 alpha^2/n)]."""
    n = X.shape[0]
    alpha = r + np.linalg.norm(X, axis=1).max()
    a = alpha ** 2 / n + 1.0
    return n * (a + np.sqrt(a ** 2 + 16.0 * alpha ** 2 / n))


def _objective_increased(candidate, current):
    rounding_tolerance = (
        32.0
        * np.finfo(float).eps
        * max(1.0, abs(candidate), abs(current))
    )
    return candidate > current + rounding_tolerance


def _step(U, V, X, rho, t, r):
    """Một bước DCA: tính (Y, Z) theo (27) rồi chiếu theo (28)."""
    D = _sq_dists(V, X)
    Y = rho * U - 2.0 * U * D + 2.0 * t * U - t
    U2 = U ** 2
    Z = rho * V - 2.0 * (V * U2.sum(axis=1, keepdims=True) - U2 @ X)
    return _proj_simplex(Y / rho), _proj_ball(Z / rho, r)


def _kmeans_plus_plus_init(X, n_clusters, rng):
    """Choose data-point centers with k-means++ distance weighting."""
    n_samples = X.shape[0]
    selected = np.empty(n_clusters, dtype=int)
    centers = np.empty((n_clusters, X.shape[1]), dtype=float)
    selected[0] = int(rng.integers(n_samples))
    centers[0] = X[selected[0]]
    closest_squared_distances = ((X - centers[0]) ** 2).sum(axis=1)

    for center_index in range(1, n_clusters):
        weights = closest_squared_distances.copy()
        weights[selected[:center_index]] = 0.0
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            available_indices = np.setdiff1d(
                np.arange(n_samples),
                selected[:center_index],
                assume_unique=True,
            )
            next_index = int(rng.choice(available_indices))
        else:
            next_index = int(rng.choice(n_samples, p=weights / weight_sum))

        selected[center_index] = next_index
        centers[center_index] = X[next_index]
        distances = ((X - centers[center_index]) ** 2).sum(axis=1)
        np.minimum(closest_squared_distances, distances, out=closest_squared_distances)

    return centers


def _centers_from_labels(X, labels, n_clusters, fallback_centers):
    counts = np.bincount(labels, minlength=n_clusters)
    centers = fallback_centers.copy()
    for feature_index in range(X.shape[1]):
        sums = np.bincount(
            labels,
            weights=X[:, feature_index],
            minlength=n_clusters,
        )
        nonempty = counts > 0
        centers[nonempty, feature_index] = sums[nonempty] / counts[nonempty]
    return centers


def _kmeans_local_search(X, labels, centers, n_clusters, steps=2):
    labels = labels.copy()
    centers = centers.copy()
    for _ in range(steps):
        centers = _centers_from_labels(X, labels, n_clusters, centers)
        labels = _sq_dists(centers, X).argmin(axis=0)
    centers = _centers_from_labels(X, labels, n_clusters, centers)
    return labels, centers


def _shake_labels(labels, n_clusters, neighborhood_size, rng):
    shaken = labels.copy()
    counts = np.bincount(shaken, minlength=n_clusters)
    changed = 0
    for sample_index in rng.permutation(len(shaken)):
        old_cluster = shaken[sample_index]
        if counts[old_cluster] <= 1:
            continue
        new_cluster = int(rng.integers(n_clusters - 1))
        if new_cluster >= old_cluster:
            new_cluster += 1
        shaken[sample_index] = new_cluster
        counts[old_cluster] -= 1
        counts[new_cluster] += 1
        changed += 1
        if changed == neighborhood_size:
            break
    return shaken


def _vns_initialization(X, initial_centers, rng, max_steps=10, max_neighborhood=5):
    """Refine a center start with bounded Hamming-neighborhood VNS."""
    n_clusters = len(initial_centers)
    initial_labels = _sq_dists(initial_centers, X).argmin(axis=0)
    best_labels, best_centers = _kmeans_local_search(
        X,
        initial_labels,
        initial_centers,
        n_clusters,
    )
    best_objective = hard_mssc_objective(X, best_labels)
    neighborhood_size = 1

    for _ in range(max_steps):
        shaken_labels = _shake_labels(
            best_labels,
            n_clusters,
            neighborhood_size,
            rng,
        )
        candidate_labels, candidate_centers = _kmeans_local_search(
            X,
            shaken_labels,
            best_centers,
            n_clusters,
        )
        candidate_objective = hard_mssc_objective(X, candidate_labels)
        if candidate_objective < best_objective:
            best_labels = candidate_labels
            best_centers = candidate_centers
            best_objective = candidate_objective
            neighborhood_size = 1
        else:
            neighborhood_size += 1
            if neighborhood_size > max_neighborhood:
                neighborhood_size = 1
    return best_centers


def _ip_dca_single(
    X,
    n_clusters,
    c_init=None,
    t=10.0,
    max_iter=1000,
    tol=1e-6,
    random_state=42,
):
    """
    Phân cụm MSSC bằng IP-DCA.

    Tham số
    -------
    X            : (n, d) dữ liệu.
    n_clusters   : số cụm c.
    t            : tham số phạt của p(U) trong (20).
    max_iter     : số vòng lặp DCA tối đa.
    tol          : dừng khi ||(U, V)^{l+1} - (U, V)^l|| <= tol.
    random_state : seed; tâm cụm khởi tạo là c điểm lấy ngẫu nhiên trong X.

    Trả về (dict)
    -------------
    labels              : (n,) nhãn cụm.
    memberships         : (c, n) ma trận thành viên U (xấp xỉ 0/1).
    centers             : (c, d) tâm cụm V.
    rho                 : giá trị rho cuối cùng.
    radius              : bán kính r của hình cầu chứa tâm cụm.
    iterations          : len(history).
    objective_history   : giá trị F1 qua các vòng lặp.
    mssc, soft_mssc     : F(U, V), giữ khóa mssc để tương thích ngược.
    hard_mssc           : MSSC của nhãn cứng, tâm được tính lại theo nhãn.
    penalty             : p(U).
    normalized_penalty  : p(U) chia cho giá trị lớn nhất trên đơn hình.
    penalized_objective : F1(U, V) = F + t * p.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    c = n_clusters
    if not 2 <= c <= n:
        raise ValueError("n_clusters phải thỏa 2 <= n_clusters <= n")

    rng = np.random.default_rng(random_state)

    # r^2 = sum_k ||x_k||^2: bán kính hình cầu chứa các tâm tối ưu
    r = np.sqrt((X ** 2).sum())

    # khởi tạo: tâm = c điểm ngẫu nhiên của dữ liệu, U = gán cứng theo tâm gần nhất
    V = X[rng.choice(n, size=c, replace=False)].copy() if c_init is None else c_init
    U = np.zeros((c, n))
    U[_sq_dists(V, X).argmin(axis=0), np.arange(n)] = 1.0

    # Bắt đầu từ cận bảo đảm tính lồi, sau đó thử giảm rho có kiểm soát.
    rho_bound = _rho_bound(X, r)
    rho = rho_bound
    history = [penalized_objective(X, U, V, t)]
    rho_reduction_steps = 0
    for _ in range(min(20, max_iter)):
        candidate_rho = rho * 0.9
        Un, Vn = _step(U, V, X, candidate_rho, t, r)
        objective = penalized_objective(X, Un, Vn, t)
        diff = np.sqrt(((Un - U) ** 2).sum() + ((Vn - V) ** 2).sum())
        if diff <= tol or _objective_increased(objective, history[-1]):
            break
        U, V = Un, Vn
        rho = candidate_rho
        rho_reduction_steps += 1
        history.append(objective)

    for _ in range(max_iter - rho_reduction_steps):
        Un, Vn = _step(U, V, X, rho, t, r)
        objective = penalized_objective(X, Un, Vn, t)
        diff = np.sqrt(((Un - U) ** 2).sum() + ((Vn - V) ** 2).sum())
        if _objective_increased(objective, history[-1]) and rho < rho_bound:
            rho = rho_bound
            Un, Vn = _step(U, V, X, rho, t, r)
            objective = penalized_objective(X, Un, Vn, t)
            diff = np.sqrt(((Un - U) ** 2).sum() + ((Vn - V) ** 2).sum())
        if _objective_increased(objective, history[-1]):
            break
        U, V = Un, Vn
        history.append(objective)
        if diff <= tol:
            break

    labels = U.argmax(axis=0)
    soft_mssc = mssc_objective(X, U, V)
    membership_penalty = penalty(U)
    return {
        "labels": labels,
        "memberships": U,
        "centers": V,
        "rho": rho,
        "rho_reduction_steps": rho_reduction_steps,
        "radius": r,
        "iterations": len(history),
        "objective_history": history,
        "mssc": soft_mssc,
        "soft_mssc": soft_mssc,
        "hard_mssc": hard_mssc_objective(X, labels),
        "penalty": membership_penalty,
        "normalized_penalty": normalized_penalty(U),
        "penalized_objective": soft_mssc + t * membership_penalty,
    }


def ip_dca(
    X,
    n_clusters,
    c_init=None,
    t=10.0,
    max_iter=1000,
    tol=1e-6,
    random_state=42,
    n_init=10,
    progress_callback: Callable[[int, int], None] | None = None,
    vns_init=True,
):
    """Run DCA from multiple k-means++ starts and retain the best objective."""
    if not isinstance(n_init, (int, np.integer)) or n_init <= 0:
        raise ValueError("n_init must be positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if t <= 0:
        raise ValueError("t must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if not isinstance(vns_init, (bool, np.bool_)):
        raise ValueError("vns_init must be a boolean")

    data = np.asarray(X, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")
    if (
        not isinstance(n_clusters, (int, np.integer))
        or not 2 <= n_clusters <= data.shape[0]
    ):
        raise ValueError("n_clusters must be an integer in [2, n_samples]")
    if c_init is not None:
        initial_centers = np.asarray(c_init, dtype=float)
        if initial_centers.shape != (n_clusters, data.shape[1]):
            raise ValueError("c_init must have shape (n_clusters, n_features)")
        if not np.isfinite(initial_centers).all():
            raise ValueError("c_init must contain only finite values")
    else:
        initial_centers = None

    rng = np.random.default_rng(random_state)
    best_result = None
    for init_index in range(n_init):
        centers = (
            initial_centers
            if init_index == 0 and initial_centers is not None
            else _kmeans_plus_plus_init(data, n_clusters, rng)
        )
        if vns_init:
            centers = _vns_initialization(data, centers, rng)
        result = _ip_dca_single(
            data,
            n_clusters,
            c_init=centers,
            t=t,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state + init_index,
        )
        if (
            best_result is None
            or result["penalized_objective"]
            < best_result["penalized_objective"]
        ):
            best_result = result
        if progress_callback is not None:
            progress_callback(init_index + 1, n_init)

    if best_result is None:
        raise RuntimeError("DCA failed to produce a clustering solution")
    best_result["vns_init"] = bool(vns_init)
    return best_result
