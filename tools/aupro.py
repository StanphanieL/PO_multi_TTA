import numpy as np
from typing import List, Tuple, Optional

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def _connected_regions_knn(xyz_pos: np.ndarray, knn: int = 16) -> np.ndarray:
    """Compute connected components among positive points using kNN graph.
    Returns an array of length P (number of positive points) with region ids [0..R-1].
    Fallback: if sklearn/scipy not available or too few points, each point is its own region.
    """
    P = xyz_pos.shape[0]
    if not SKLEARN_OK or not SCIPY_OK or P < 2 or knn < 1:
        return np.arange(P, dtype=np.int32)
    k = min(knn, P)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xyz_pos)
    inds = nbrs.kneighbors(xyz_pos, return_distance=False)
    # build symmetric adjacency
    rows = np.repeat(np.arange(P), k)
    cols = inds.reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.uint8)
    A = csr_matrix((data, (rows, cols)), shape=(P, P))
    A = A.maximum(A.transpose())
    n_comp, labels = connected_components(A, directed=False)
    return labels.astype(np.int32)


def aupro_curve(
    scores_list: List[np.ndarray],
    gts_list: List[np.ndarray],
    xyz_list: Optional[List[np.ndarray]] = None,
    fpr_max: float = 0.3,
    num_fpr: int = 31,
    region_knn: int = 16,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute PRO-vs-FPR curve and AUPRO across samples.

    Args:
        scores_list: list of (N_i,) arrays of point-level anomaly scores per sample (larger=more anomalous).
        gts_list:    list of (N_i,) binary arrays (0/1) of ground-truth anomaly labels per sample.
        xyz_list:    optional list of (N_i,3) coordinates for region connectivity; if None, each positive point is a region.
        fpr_max:     consider FPR in [0, fpr_max].
        num_fpr:     number of FPR grid points (inclusive), e.g., 31 → 0, 0.01, ..., 0.3.
        region_knn:  k for kNN region connectivity among positive points.

    Returns:
        fpr_grid:    (M,) FPR grid.
        pro_curve:   (M,) mean PRO at each FPR.
        aupro_area:  trapezoidal area under PRO-FPR in [0, fpr_max].
        aupro_norm:  normalized area = area / fpr_max.
    """
    assert len(scores_list) == len(gts_list) and len(scores_list) > 0
    # build thresholds by negatives' quantiles
    neg_scores = []
    for s, g in zip(scores_list, gts_list):
        s = np.asarray(s).reshape(-1)
        g = np.asarray(g).reshape(-1)
        if (g == 0).any():
            neg_scores.append(s[g == 0])
    if len(neg_scores) == 0:
        # fallback: cannot estimate thresholds; return zeros
        fpr_grid = np.linspace(0.0, fpr_max, num_fpr)
        pro = np.zeros_like(fpr_grid)
        return fpr_grid, pro, 0.0, 0.0
    neg_all = np.concatenate(neg_scores, axis=0)
    fpr_grid = np.linspace(0.0, fpr_max, num_fpr)
    # avoid exactly 1.0 quantile
    q = np.clip(1.0 - fpr_grid, 0.0, 1.0)
    thresholds = np.quantile(neg_all, q, method='nearest') if hasattr(np, 'quantile') else np.percentile(neg_all, q * 100.0)

    # precompute region labels per sample
    regions_per_sample = []
    for idx, (s, g) in enumerate(zip(scores_list, gts_list)):
        g = np.asarray(g).reshape(-1)
        if g.sum() == 0:
            regions_per_sample.append(None)
            continue
        if xyz_list is None or xyz_list[idx] is None:
            # each positive point becomes its own region
            labels_pos = np.arange(int(g.sum()), dtype=np.int32)
        else:
            xyz = np.asarray(xyz_list[idx])
            labels_pos = _connected_regions_knn(xyz[g == 1], knn=region_knn)
        regions_per_sample.append(labels_pos)

    # evaluate PRO at each threshold
    pro_vals = []
    for t in thresholds:
        pros = []
        for (s, g, reg) in zip(scores_list, gts_list, regions_per_sample):
            s = np.asarray(s).reshape(-1)
            g = np.asarray(g).reshape(-1)
            if g.sum() == 0:
                continue
            pred = (s >= t)
            # gather positive indices and their region ids
            pos_idx = np.where(g == 1)[0]
            if reg is None:
                # each positive point is its own region
                overlap = pred[pos_idx].astype(np.float32)
                pro_sample = float(overlap.mean())
            else:
                labels = reg
                R = labels.max() + 1 if labels.size > 0 else 0
                if R == 0:
                    continue
                pro_regions = []
                for r in range(R):
                    ridx = pos_idx[labels == r]
                    if ridx.size == 0:
                        continue
                    pro_regions.append(pred[ridx].mean())
                if len(pro_regions) == 0:
                    continue
                pro_sample = float(np.mean(pro_regions))
            pros.append(pro_sample)
        pro_vals.append(np.mean(pros) if len(pros) > 0 else 0.0)
    pro_curve = np.array(pro_vals)
    # area under curve over [0, fpr_max]
    aupro_area = float(np.trapz(pro_curve, fpr_grid))
    aupro_norm = aupro_area / max(fpr_max, 1e-12)
    return fpr_grid, pro_curve, aupro_area, aupro_norm