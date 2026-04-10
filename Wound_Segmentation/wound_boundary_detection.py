import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


# ==============================================================================
# 1. SCALE NORMALIZATION
# ==============================================================================

def normalize_point_cloud(xyz):
    """
    Translate to centroid and scale to unit sphere.
    Ensures curvature estimates are scale-invariant across samples.
    Without this, curvature magnitudes vary with scanner physical units,
    making any threshold meaningless across samples.
    """
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    scale = np.linalg.norm(xyz, axis=1).max() + 1e-8
    return xyz / scale


# ==============================================================================
# 2. TRUE PRINCIPAL CURVATURE via LOCAL QUADRATIC SURFACE FITTING
# ==============================================================================

def compute_true_curvature(xyz, k=30):
    """
    Compute genuine principal curvatures by fitting a local quadratic patch
    and extracting the shape operator (second fundamental form).

    Why not covariance eigenvalues (original code)?
    Covariance eigenvalues measure local point *spread*, not curvature.
    A flat but noisy patch scores high; a smooth wound edge scores low.
    Quadratic fitting gives geometrically correct k1, k2.

    Returns
    -------
    curvedness      : [N]  sqrt((k1^2 + k2^2) / 2)
    mean_curvature  : [N]  (k1 + k2) / 2
    shape_index     : [N]  (2/pi) * arctan((k1+k2) / (|k1|+|k2|))
    normals         : [N, 3]
    """
    N = len(xyz)
    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k)

    k1_all  = np.zeros(N, dtype=np.float64)
    k2_all  = np.zeros(N, dtype=np.float64)
    normals = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        nbr      = xyz[indices[i]]          # [k, 3]
        centered = nbr - nbr.mean(axis=0)

        # Step 1: PCA → local tangent frame
        cov              = centered.T @ centered / k
        eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
        normal           = eigvecs[:, 0]          # min eigval → normal
        u_ax             = eigvecs[:, 2]          # max eigval → tangent u
        v_ax             = eigvecs[:, 1]          # mid eigval → tangent v
        normals[i]       = normal

        # Step 2: Project neighbors into local UV frame
        us = centered @ u_ax   # [k]
        vs = centered @ v_ax   # [k]
        ws = centered @ normal  # [k] height above tangent plane

        # Step 3: Fit quadratic  w = au² + buv + cv² + du + ev
        A = np.column_stack([us**2, us * vs, vs**2, us, vs])  # [k, 5]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, ws, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b, c, d, e = coeffs

        # Step 4: Shape operator at origin (orthonormal frame → E=G=1, F=0)
        #   [[2a,  b],
        #    [ b, 2c]]
        eigvals_s  = np.linalg.eigvalsh(np.array([[2*a, b], [b, 2*c]]))
        k1_all[i]  = eigvals_s[1]   # max principal curvature
        k2_all[i]  = eigvals_s[0]   # min principal curvature

    curvedness     = np.sqrt((k1_all**2 + k2_all**2) / 2.0)
    mean_curvature = (k1_all + k2_all) / 2.0
    denom          = np.abs(k1_all) + np.abs(k2_all) + 1e-8
    shape_index    = (2.0 / np.pi) * np.arctan((k1_all + k2_all) / denom)

    return curvedness, mean_curvature, shape_index, normals


# ==============================================================================
# 3. NORMAL DISCONTINUITY
# ==============================================================================

def compute_normal_discontinuity(normals, xyz, k=16):
    """
    Mean angular deviation of surface normals across each k-neighborhood.
    High discontinuity marks the wound boundary, where normals swing sharply
    from the wound-bed orientation to the surrounding skin orientation.
    This is an independent and complementary cue to curvedness.
    """
    tree    = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k)

    nbr_normals    = normals[indices]              # [N, k, 3]
    center_normals = normals[:, np.newaxis, :]     # [N, 1, 3]

    dot    = np.einsum('nkd,nkd->nk',
                       np.broadcast_to(center_normals, nbr_normals.shape),
                       nbr_normals)
    dot    = np.clip(dot, -1.0, 1.0)
    angles = np.arccos(np.abs(dot))                # [N, k]
    return angles.mean(axis=1)                     # [N]


# ==============================================================================
# 4. COMBINED SALIENCY SCORE  (stays SOFT — continuous in [0, 1])
# ==============================================================================

def combined_saliency(curvedness, mean_curvature, normal_discontinuity):
    """
    Fuse geometric cues into a single soft score in [0, 1].

    Kept soft deliberately: hard thresholding here discards confidence
    information that the downstream GNN can exploit.  A point with
    saliency 0.95 and one with 0.51 are geometrically very different;
    treating them identically would be wasteful.
    """
    def norm01(x):
        rng = x.max() - x.min() + 1e-8
        return (x - x.min()) / rng

    c  = norm01(curvedness)
    mc = norm01(np.abs(mean_curvature))
    nd = norm01(normal_discontinuity)

    return 0.4 * c + 0.3 * mc + 0.3 * nd   # [N], continuous in [0,1]


# ==============================================================================
# 5. SOFT SEED via GMM  (returns PROBABILITIES, not a binary mask)
# ==============================================================================

def soft_seed_gmm(saliency, n_components=2):
    """
    Fit a 2-component GMM to the saliency distribution and return the
    posterior probability of belonging to the HIGH-saliency component.

    Why soft, not hard?
    ------------------
    The GNN (section 6) uses this as a continuous input feature.
    Hard binarization at this stage would throw away gradient information
    that the GNN needs to propagate saliency correctly across the graph.

    Falls back to a rank-normalised score if GMM fitting fails.

    Returns
    -------
    p_wound : [N]  soft wound probability in (0, 1)
    """
    X = saliency.reshape(-1, 1).astype(np.float64)
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42,
                              max_iter=200, n_init=3)
        gmm.fit(X)

        # Identify which component has the higher mean (= wound component)
        wound_component = int(np.argmax(gmm.means_.flatten()))
        p_wound = gmm.predict_proba(X)[:, wound_component]  # [N], soft

        # Sanity-check: wound component should cover a plausible minority
        high_conf = (p_wound > 0.5).mean()
        if not (0.05 < high_conf < 0.70):
            raise ValueError(f"GMM gave implausible wound fraction {high_conf:.2f}")

        return p_wound

    except Exception:
        # Fallback: rank-normalise the raw saliency score
        ranks   = saliency.argsort().argsort().astype(np.float64)
        p_wound = ranks / (len(ranks) - 1 + 1e-8)
        return p_wound


# ==============================================================================
# 6. SHALLOW GNN — saliency propagation on sparse k-NN graph
# ==============================================================================

def build_knn_graph(xyz, k=16):
    """
    Build a sparse k-NN adjacency structure.
    Returns indices [N, k] and edge weights [N, k] based on Gaussian distance.
    """
    tree        = cKDTree(xyz)
    dists, indices = tree.query(xyz, k=k)

    # Gaussian kernel weights: closer neighbors have more influence
    sigma       = np.median(dists[:, 1:]) + 1e-8   # robust bandwidth
    weights     = np.exp(-(dists**2) / (2 * sigma**2))   # [N, k]
    weights    /= weights.sum(axis=1, keepdims=True) + 1e-8  # row-normalise

    return indices, weights


def gnn_propagate(p_wound, node_features, indices, weights, n_layers=3):
    """
    Shallow, geometry-aware message-passing GNN.

    Design rationale
    ----------------
    - Kept to 3 layers to avoid over-smoothing on small datasets (~70 scans).
      Deep GNNs homogenise node representations; the geometry features must
      remain the dominant signal, with the GNN acting as a learned smoother.
    - No learnable parameters: uses fixed geometry-weighted aggregation.
      This is intentional at this data scale — a parameterised GNN would
      overfit badly before Aim 1 data collection matures.  Parameters can
      be introduced once labelled data exceeds ~500 scans.
    - Edge weighting encodes *both* spatial proximity (Gaussian distance
      weight) and geometric similarity (feature-space agreement), so the
      graph respects surface topology, not just Euclidean proximity.

    Message passing rule (one layer)
    ---------------------------------
        h_i ← (1 - alpha) * h_i  +  alpha * Σ_j  w_ij * geo_ij * h_j

    where:
        h_i       = current node saliency
        w_ij      = Gaussian spatial weight (from build_knn_graph)
        geo_ij    = geometric agreement between node i and neighbour j
        alpha     = propagation strength (0.4 — preserves seed signal)

    Parameters
    ----------
    p_wound       : [N]     soft GMM wound probability (seed signal)
    node_features : [N, F]  per-point geometry features for edge weighting
    indices       : [N, k]  k-NN neighbour indices
    weights       : [N, k]  Gaussian spatial edge weights
    n_layers      : int     number of message-passing steps

    Returns
    -------
    h : [N]  propagated wound probability, still soft in [0, 1]
    """
    N, k    = indices.shape
    alpha   = 0.4        # propagation strength; retains seed signal at 60%

    # Normalise node features for geometric agreement computation
    nf_norm = node_features / (
        np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8
    )

    h = p_wound.copy()   # initialise from soft GMM seed

    for _ in range(n_layers):
        nbr_h   = h[indices]                          # [N, k]  neighbour values

        # Geometric agreement: cosine similarity in feature space
        nbr_nf  = nf_norm[indices]                    # [N, k, F]
        geo_sim = np.einsum('nf,nkf->nk',
                            nf_norm, nbr_nf)          # [N, k] in [-1, 1]
        geo_sim = (geo_sim + 1.0) / 2.0               # rescale to [0, 1]

        # Combined edge weight: spatial × geometric
        edge_w  = weights * geo_sim                   # [N, k]
        edge_w /= edge_w.sum(axis=1, keepdims=True) + 1e-8

        # Message aggregation
        agg     = (edge_w * nbr_h).sum(axis=1)        # [N]

        # Update: retain seed signal, blend in neighbourhood message
        h       = (1.0 - alpha) * h + alpha * agg

    # Final soft clamp to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    return h


# ==============================================================================
# 7. REGION MASK → BOUNDARY RING  (region-then-boundary, not edge classification)
# ==============================================================================

def majority_vote_smooth(mask, indices, k=16, passes=2, threshold=0.6):
    """
    Iterative majority-vote smoothing on the k-NN graph.
    Removes isolated noise points before boundary extraction.
    Operating on the filled region first makes the boundary topologically
    robust: a single misclassified point cannot break the boundary ring,
    unlike direct edge classification.
    """
    smoothed = mask.copy().astype(np.float64)
    for _ in range(passes):
        nbr_labels = smoothed[indices]             # [N, k]
        vote       = nbr_labels.mean(axis=1)       # fraction of wound neighbours
        smoothed   = (vote > threshold).astype(np.float64)
    return smoothed.astype(np.int64)


def extract_boundary_ring(region_mask, indices):
    """
    Keep only points on the interface between wound and healthy regions.

    A point is a boundary point iff:
      - it is labelled wound (region_mask == 1), AND
      - at least one neighbour is labelled healthy (region_mask == 0)

    Why region-then-boundary rather than direct edge classification?
    ---------------------------------------------------------------
    Edge classification on sparse graphs is noisy: a single bad edge
    prediction breaks the boundary ring topology.  Predicting a filled
    region first and extracting the interface is more robust: the region
    acts as a topological prior, and the boundary extraction is
    deterministic and cannot produce disconnected fragments from a
    single misclassification.
    """
    nbr_masks   = region_mask[indices]              # [N, k]
    has_wound   = nbr_masks.max(axis=1) == 1
    has_healthy = nbr_masks.min(axis=1) == 0

    boundary = np.zeros(len(region_mask), dtype=np.int64)
    boundary[(region_mask == 1) & has_wound & has_healthy] = 1
    return boundary


# ==============================================================================
# 8. MAIN PIPELINE
# ==============================================================================

def apply_wound_boundary_detection(data_dir, k_curvature=30, k_graph=16):
    """
    Full pipeline — saliency seeding stage for C.3.1.

    Steps
    -----
    1.  Scale-normalise point cloud                  → scale invariance
    2.  True principal curvatures (quadratic fit)    → k1, k2, shape_index
    3.  Normal discontinuity                         → surface orientation cue
    4.  Combined soft saliency score                 → continuous [0,1]
    5.  Soft GMM seed                                → p_wound per point
    6.  Shallow GNN propagation on k-NN graph        → spatially coherent p_wound
    7.  Binarise propagated score                    → filled region mask
    8.  Majority-vote smoothing                      → noise removal
    9.  Boundary ring extraction                     → final interface mask
    10. Save mask + intermediate features            → HDF5

    What is deliberately NOT included and why
    ------------------------------------------
    - No learned shape prior: requires calibrated Aim 1 annotations that
      do not yet exist.  Will be added once labelled data exceeds ~500 scans.
    - No parameterised GNN weights: would overfit at current data scale.
      The GNN here acts as a geometry-aware smoother, not a classifier.
    - No direct edge classification for boundary prediction: topologically
      fragile on sparse graphs; region-then-boundary is more robust.
    """
    splits = ['train', 'test']

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print("path doesn't exist")
            continue

        print(f"\nProcessing {split} split...")
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                print("path is not dir")
                continue

            files = [f for f in os.listdir(class_dir) if f.endswith('.h5')]
            for filename in tqdm(files, desc=f"Class: {class_name}"):
                filepath = os.path.join(class_dir, filename)
                df       = pd.read_hdf(filepath, 'df')

                # if 'mask' in df.columns:
                #     print("Already has masks")
                #     continue

                # ----------------------------------------------------------
                # 1. Normalise
                # ----------------------------------------------------------
                xyz_raw = df[['x', 'y', 'z']].values.astype(np.float64)
                xyz     = normalize_point_cloud(xyz_raw)

                # ----------------------------------------------------------
                # 2. Geometry features
                # ----------------------------------------------------------
                curvedness, mean_curvature, shape_index, normals = \
                    compute_true_curvature(xyz, k=k_curvature)

                normal_disc = compute_normal_discontinuity(
                    normals, xyz, k=k_graph)

                # ----------------------------------------------------------
                # 3. Soft saliency (stays continuous — fed into GNN)
                # ----------------------------------------------------------
                saliency = combined_saliency(
                    curvedness, mean_curvature, normal_disc)

                # ----------------------------------------------------------
                # 4. GMM soft seed  (probability, not binary)
                # ----------------------------------------------------------
                p_wound = soft_seed_gmm(saliency)

                # ----------------------------------------------------------
                # 5. Build k-NN graph once; reuse for GNN + smoothing
                # ----------------------------------------------------------
                indices, weights = build_knn_graph(xyz, k=k_graph)

                # ----------------------------------------------------------
                # 6. GNN propagation — geometry-weighted message passing
                #    Node features: stack all geometry cues
                # ----------------------------------------------------------
                node_features = np.column_stack([
                    curvedness,
                    np.abs(mean_curvature),
                    shape_index,
                    normal_disc,
                    saliency,
                ])                                          # [N, 5]

                p_propagated = gnn_propagate(
                    p_wound, node_features, indices, weights, n_layers=3)

                # ----------------------------------------------------------
                # 7. Binarise propagated score
                #    Threshold at 0.5: the GNN output is a calibrated
                #    probability so 0.5 is the natural decision boundary.
                # ----------------------------------------------------------
                region_mask = (p_propagated > 0.5).astype(np.int64)

                # ----------------------------------------------------------
                # 8. Smooth region mask (majority vote on pre-built graph)
                # ----------------------------------------------------------
                region_mask = majority_vote_smooth(
                    region_mask, indices, k=k_graph, passes=2)

                # ----------------------------------------------------------
                # 9. Extract boundary ring
                # ----------------------------------------------------------
                boundary_mask = extract_boundary_ring(region_mask, indices)

                # ----------------------------------------------------------
                # 10. Save
                # ----------------------------------------------------------
                df['mask']            = boundary_mask
                df['p_wound']         = p_propagated   # soft, for GNN training
                df['saliency']        = saliency
                df['curvedness']      = curvedness
                df['mean_curvature']  = mean_curvature
                df['shape_index']     = shape_index
                df['normal_disc']     = normal_disc
                df.to_hdf(filepath, key='df', mode='w')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    DATA_DIR = "D:/MD_Implementations_copy/sorted_data"
    apply_wound_boundary_detection(
        DATA_DIR,
        k_curvature=30,   # neighbours for quadratic surface fit
        k_graph=16,       # neighbours for GNN graph + smoothing + boundary
    )
    print("\nWound boundary masks generated successfully!")
