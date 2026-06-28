"""
Numba in-process equivalent of dbscan_main.cpp.

Same algorithm as the C++ tool (anisotropic spatial/ToF DBSCAN per shot,
border points included via BFS, weighted centroid aggregation) but called
directly as a Python function on in-memory arrays — no subprocess, no
intermediate files. See dbscan_main.cpp for the original and the discussion
of why a subprocess-based tool isn't a good fit for live use; this module
exists to let the same algorithm be benchmarked/used in-process.

This first cut deliberately keeps the same O(n^2) per-shot neighbor search
as the C++ version, so timing comparisons isolate the execution-model
difference (in-process JIT vs. subprocess+file I/O) rather than mixing in
an algorithmic complexity change.
"""

import numpy as np
from numba import njit, prange

from SERVAL.core.data_types import EVENT_DTYPE
from SERVAL.postprocessing.centroiding import CENTROID_DTYPE


@njit(cache=True)
def _dbscan_shot(x, y, tof, epsilon, eps_time, min_points):
    """
    Cluster one shot's points. Same semantics as dbscan_main.cpp's
    dbscan()/expandCluster(): a point with >= min_points neighbors (spatial
    distance <= epsilon AND |tof diff| <= eps_time) is a core point and pulls
    its neighbors into the cluster (including non-core "border" points,
    which join but don't themselves expand).

    Returns
    -------
    labels : np.ndarray[int64]
        -1 = noise, otherwise a 0-based cluster id local to this shot.
    n_clusters : int
    """
    n = len(x)
    label = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    eps2 = epsilon * epsilon

    neighbor_buf = np.empty(n, dtype=np.int64)
    queue_buf = np.empty(n, dtype=np.int64)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue

        xi, yi, ti = x[i], y[i], tof[i]
        cnt = 0
        for j in range(n):
            if j == i:
                continue
            if abs(tof[j] - ti) > eps_time:
                continue
            dx = x[j] - xi
            dy = y[j] - yi
            if dx * dx + dy * dy <= eps2:
                neighbor_buf[cnt] = j
                cnt += 1

        if cnt < min_points:
            visited[i] = True
            label[i] = -1
            continue

        visited[i] = True
        label[i] = cluster_id
        qhead = 0
        qtail = 1
        queue_buf[0] = i

        while qhead < qtail:
            cur = queue_buf[qhead]
            qhead += 1

            xc, yc, tc = x[cur], y[cur], tof[cur]
            cnt2 = 0
            for j in range(n):
                if j == cur:
                    continue
                if abs(tof[j] - tc) > eps_time:
                    continue
                dx = x[j] - xc
                dy = y[j] - yc
                if dx * dx + dy * dy <= eps2:
                    neighbor_buf[cnt2] = j
                    cnt2 += 1

            if cnt2 < min_points:
                continue  # border point: joined, but does not expand

            for k in range(cnt2):
                nb = neighbor_buf[k]
                if not visited[nb]:
                    visited[nb] = True
                    label[nb] = cluster_id
                    queue_buf[qtail] = nb
                    qtail += 1
                elif label[nb] == -1:
                    label[nb] = cluster_id

        cluster_id += 1

    return label, cluster_id


@njit(cache=True, parallel=True)
def _cluster_all_shots(x, y, tof, shot_starts, epsilon, eps_time, min_points):
    """
    Run _dbscan_shot independently per shot (parallel across shots).

    Parameters
    ----------
    x, y, tof : np.ndarray
        Already filtered (ToF window) and sorted by shot (t_trigger).
    shot_starts : np.ndarray[int64]
        Boundaries into x/y/tof: shot s spans [shot_starts[s], shot_starts[s+1]).

    Returns
    -------
    all_labels : np.ndarray[int64]
        Per-point cluster label, local to its shot (-1 = noise).
    n_clusters_per_shot : np.ndarray[int64]
    """
    n_shots = len(shot_starts) - 1
    all_labels = np.empty(len(x), dtype=np.int64)
    n_clusters_per_shot = np.zeros(n_shots, dtype=np.int64)

    for s in prange(n_shots):
        start = shot_starts[s]
        end = shot_starts[s + 1]
        labels, n_clusters = _dbscan_shot(
            x[start:end], y[start:end], tof[start:end], epsilon, eps_time, min_points
        )
        all_labels[start:end] = labels
        n_clusters_per_shot[s] = n_clusters

    return all_labels, n_clusters_per_shot


def centroid_events(
    events: np.ndarray,
    epsilon: float = 2.0,
    eps_time: float = 100e-9,
    min_points: int = 1,
    tof_min: float = 0.0,
    tof_max: float = 1.0,
    correction_tof_vals: np.ndarray = None,
    correction_vals: np.ndarray = None,
) -> np.ndarray:
    """
    Cluster an EVENT_DTYPE array into weighted centroids (CENTROID_DTYPE).

    Same parameters and semantics as CentroidProcessor / dbscan_main.cpp:
    epsilon (pixels) and eps_time (seconds) are independent criteria for two
    points to be neighbors; tof_min/tof_max is a pre-filter window applied
    before clustering.
    """
    if len(events) == 0:
        return np.array([], dtype=CENTROID_DTYPE)

    tof = events["tof"]
    mask = (tof >= tof_min) & (tof <= tof_max)
    ev = events[mask]
    if len(ev) == 0:
        return np.array([], dtype=CENTROID_DTYPE)

    tof = ev["tof"].copy()
    if correction_tof_vals is not None and len(correction_tof_vals):
        # np.interp clamps to the edge y-value for out-of-range x, matching
        # dbscan_main.cpp's interpolate() behaviour exactly.
        tof = tof - np.interp(tof, correction_tof_vals, correction_vals)

    # Group by shot: sort by t_trigger, then find each unique value's run.
    order = np.argsort(ev["t_trigger"], kind="stable")
    t_trigger = ev["t_trigger"][order]
    x = ev["x"][order].astype(np.float64)
    y = ev["y"][order].astype(np.float64)
    tof = tof[order]
    tot = ev["tot"][order].astype(np.float64)

    shot_vals, shot_start_idx = np.unique(t_trigger, return_index=True)
    shot_starts = np.append(shot_start_idx, len(ev)).astype(np.int64)

    all_labels, n_clusters_per_shot = _cluster_all_shots(
        x, y, tof, shot_starts, float(epsilon), float(eps_time), int(min_points)
    )

    n_shots = len(shot_vals)
    total_clusters = int(n_clusters_per_shot.sum())
    if total_clusters == 0:
        return np.array([], dtype=CENTROID_DTYPE)

    # Map each point's per-shot-local label to a globally unique cluster id.
    shot_offsets = np.zeros(n_shots, dtype=np.int64)
    shot_offsets[1:] = np.cumsum(n_clusters_per_shot)[:-1]
    point_shot_idx = np.repeat(np.arange(n_shots), np.diff(shot_starts))

    valid = all_labels != -1
    global_cluster_id = all_labels[valid] + shot_offsets[point_shot_idx[valid]]

    sum_x = np.zeros(total_clusters, dtype=np.float64)
    sum_y = np.zeros(total_clusters, dtype=np.float64)
    sum_tof = np.zeros(total_clusters, dtype=np.float64)
    sum_w = np.zeros(total_clusters, dtype=np.float64)
    max_tot = np.zeros(total_clusters, dtype=np.float64)

    w = tot[valid]
    np.add.at(sum_x, global_cluster_id, x[valid] * w)
    np.add.at(sum_y, global_cluster_id, y[valid] * w)
    np.add.at(sum_tof, global_cluster_id, tof[valid] * w)
    np.add.at(sum_w, global_cluster_id, w)
    np.maximum.at(max_tot, global_cluster_id, w)

    # Each cluster's t_trigger is its shot's trigger time.
    cluster_shot_idx = np.repeat(np.arange(n_shots), n_clusters_per_shot)
    cluster_t_trigger = shot_vals[cluster_shot_idx]

    centroids = np.empty(total_clusters, dtype=CENTROID_DTYPE)
    centroids["t_trigger"] = cluster_t_trigger
    centroids["x"] = sum_x / sum_w
    centroids["y"] = sum_y / sum_w
    centroids["tof"] = sum_tof / sum_w
    centroids["tot"] = max_tot
    return centroids


def centroid_pixels_dbscan(
    x: np.ndarray,
    y: np.ndarray,
    toa: np.ndarray,
    tot: np.ndarray,
    trigger_times: np.ndarray = None,
    epsilon: float = 2.0,
    eps_time: float = 100e-9,
    min_points: int = 1,
):
    """
    Used by ExtractorWorker for live (pre-correlation) raw-pixel centroiding:
    out_x, out_y, out_toa, out_tot, n_out — same return contract as the
    greedy buffer-based algorithm this replaced.

    Unlike centroid_events() (which clusters already-correlated, per-shot
    EVENT_DTYPE records by ToF), this clusters raw PixelData by (x, y, toa)
    BEFORE trigger correlation — there is no t_trigger/tof yet, and no
    shot grouping inherent to the data.

    Performance: charge-sharing clusters only ever span a few ns, far less
    than the gap between triggers (shot period), so splitting the chunk into
    per-shot groups using `trigger_times` (if given) before running the O(n^2)
    neighbor search loses essentially nothing while turning one large
    O(N^2) search into many small O(n_shot^2) ones — much cheaper in total
    for the same reason this already mattered for the offline per-shot
    centroiding. Pass `trigger_times=None` to disable this (single group,
    matches the unsplit cost of the naive approach).

    Unlike centroid_events()/dbscan_main.cpp, isolated hits (DBSCAN "noise",
    i.e. fewer than min_points neighbors) are NOT dropped — every real pixel
    hit must still be reported, charge-sharing cleanup just means merging
    genuinely-split hits, not discarding lone ones.

    Returns
    -------
    out_x, out_y : np.ndarray[uint16]
        Centroid coordinates (position of the max-ToT hit in each cluster).
    out_toa : np.ndarray[float64]
        Centroid ToA (minimum ToA in cluster), seconds.
    out_tot : np.ndarray[uint32]
        Maximum ToT in cluster.
    n_out : int
        Number of centroids produced (== len(out_x)).
    """
    n = len(x)
    if n == 0:
        return (
            np.empty(0, dtype=np.uint16), np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint32), 0,
        )

    order = np.argsort(toa, kind="stable")
    xs = x[order].astype(np.float64)
    ys = y[order].astype(np.float64)
    toas = toa[order]
    tots = tot[order].astype(np.float64)

    if trigger_times is not None and len(trigger_times) > 1:
        # Assign each (sorted) hit to the shot whose trigger most recently
        # preceded it — same binary-search rule used for event correlation.
        shot_idx = np.clip(np.searchsorted(trigger_times, toas, side="right") - 1, 0, None)
        # shot_idx is non-decreasing since both toas and trigger_times are
        # sorted; unique() on a non-decreasing array gives contiguous runs.
        _, shot_start_idx = np.unique(shot_idx, return_index=True)
        shot_starts = np.append(shot_start_idx, n).astype(np.int64)
    else:
        shot_starts = np.array([0, n], dtype=np.int64)

    all_labels, n_clusters_per_shot = _cluster_all_shots(
        xs, ys, toas, shot_starts, float(epsilon), float(eps_time), int(min_points)
    )

    n_shots = len(shot_starts) - 1
    total_clusters = int(n_clusters_per_shot.sum())
    shot_offsets = np.zeros(n_shots, dtype=np.int64)
    shot_offsets[1:] = np.cumsum(n_clusters_per_shot)[:-1]
    point_shot_idx = np.repeat(np.arange(n_shots), np.diff(shot_starts))

    is_clustered = all_labels != -1
    global_cluster_id = all_labels[is_clustered] + shot_offsets[point_shot_idx[is_clustered]]

    # Representative position = the max-ToT hit in each cluster: sort by
    # (cluster, -tot) so each cluster's first row after sorting is its max.
    clustered_idx = np.flatnonzero(is_clustered)
    order2 = np.lexsort((-tots[clustered_idx], global_cluster_id))
    sorted_cluster_id = global_cluster_id[order2]
    _, first_idx = np.unique(sorted_cluster_id, return_index=True)
    rep_idx = clustered_idx[order2[first_idx]]

    cluster_x = xs[rep_idx]
    cluster_y = ys[rep_idx]
    cluster_tot = tots[rep_idx]

    cluster_min_toa = np.full(total_clusters, np.inf, dtype=np.float64)
    np.minimum.at(cluster_min_toa, global_cluster_id, toas[clustered_idx])

    # Noise points are never dropped: each becomes its own singleton cluster.
    noise_idx = np.flatnonzero(~is_clustered)
    n_total = total_clusters + len(noise_idx)

    out_x = np.empty(n_total, dtype=np.uint16)
    out_y = np.empty(n_total, dtype=np.uint16)
    out_toa = np.empty(n_total, dtype=np.float64)
    out_tot = np.empty(n_total, dtype=np.uint32)

    out_x[:total_clusters] = cluster_x.astype(np.uint16)
    out_y[:total_clusters] = cluster_y.astype(np.uint16)
    out_toa[:total_clusters] = cluster_min_toa
    out_tot[:total_clusters] = cluster_tot.astype(np.uint32)

    out_x[total_clusters:] = xs[noise_idx].astype(np.uint16)
    out_y[total_clusters:] = ys[noise_idx].astype(np.uint16)
    out_toa[total_clusters:] = toas[noise_idx]
    out_tot[total_clusters:] = tots[noise_idx].astype(np.uint32)

    return out_x, out_y, out_toa, out_tot, n_total


def process_file(
    events_dat_path,
    epsilon: float = 2.0,
    eps_time: float = 100e-9,
    min_points: int = 1,
    tof_min: float = 0.0,
    tof_max: float = 1.0,
    correction_path=None,
) -> np.ndarray:
    """Load an _events.dat file and return its CENTROID_DTYPE centroids."""
    events = np.fromfile(str(events_dat_path), dtype=EVENT_DTYPE)

    correction_tof_vals = correction_vals = None
    if correction_path is not None:
        data = np.loadtxt(correction_path, delimiter=",")
        correction_tof_vals, correction_vals = data[:, 0], data[:, 1]

    return centroid_events(
        events,
        epsilon=epsilon,
        eps_time=eps_time,
        min_points=min_points,
        tof_min=tof_min,
        tof_max=tof_max,
        correction_tof_vals=correction_tof_vals,
        correction_vals=correction_vals,
    )
