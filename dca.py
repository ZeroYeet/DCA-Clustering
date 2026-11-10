from collections.abc import Callable

import numpy as np


def _sq_dists(V, X):
    """D[i, k] = ||x_k - v_i||^2, kích thước (c, n)."""
    return ((V[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)


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


def penalty(U):
    """p(U) = sum_i sum_k u_ik (1 - u_ik); bằng 0 khi U là ma trận 0/1."""
    return float((U * (1.0 - U)).sum())


def penalized_objective(X, U, V, t):
    """F1(U, V) = F(U, V) + t * p(U)   (công thức (20))."""
    return mssc_objective(X, U, V) + t * penalty(U)


def _rho_bound(X, r):
    """Công thức (24): rho >= n [ (xi^2/n + 1) + sqrt((xi^2/n + 1)^2 + 12 xi^2 / n) ]."""
    n = X.shape[0]
    xi = r + np.linalg.norm(X, axis=1).max()
    a = xi ** 2 / n + 1.0
    return n * (a + np.sqrt(a ** 2 + 12.0 * xi ** 2 / n))


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
    mssc                : F(U, V).
    penalty             : p(U).
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

    # rho khởi tạo theo (24), rồi giảm dần (thủ tục "Update rho", mục 2.3)
    rho_max = _rho_bound(X, r)
    rho, tau = rho_max, 0.9
    history = [penalized_objective(X, U, V, t)]
    warmup = 0
    for _ in range(min(200, max_iter)):
        Un, Vn = _step(U, V, X, tau * rho, t, r)
        f_new = penalized_objective(X, Un, Vn, t)
        if f_new > history[-1]:
            break
        U, V, rho = Un, Vn, tau * rho
        history.append(f_new)
        warmup += 1

    n_iter = warmup
    for n_iter in range(warmup + 1, max_iter + 1):
        Un, Vn = _step(U, V, X, rho, t, r)
        # nếu F1 tăng thì rho quá nhỏ (H1 mất tính lồi): tăng rho rồi thử lại
        tries = 0
        while penalized_objective(X, Un, Vn, t) > history[-1] + 1e-12 and rho < rho_max and tries < 100:
            rho = min(rho / tau, rho_max)
            Un, Vn = _step(U, V, X, rho, t, r)
            tries += 1
        diff = np.sqrt(((Un - U) ** 2).sum() + ((Vn - V) ** 2).sum())
        U, V = Un, Vn
        history.append(penalized_objective(X, U, V, t))
        if diff <= tol:
            break

    labels = U.argmax(axis=0)
    return {
        "labels": labels,
        "memberships": U,
        "centers": V,
        "rho": rho,
        "radius": r,
        "iterations": len(history),
        "objective_history": history,
        "mssc": mssc_objective(X, U, V),
        "penalty": penalty(U),
        "penalized_objective": penalized_objective(X, U, V, t),
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
    return best_result
