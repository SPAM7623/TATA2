"""TASK 2: POSITIVE CLUSTERING ANALYSIS
Determine if 66 defect samples form multiple hidden clusters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import pdist, squareform

from src import config, data


def analyze_positive_clustering():
    """Cluster defect samples to find hidden defect types."""
    print("=" * 80)
    print("TASK 2: POSITIVE CLUSTERING ANALYSIS")
    print("=" * 80)

    # Load data
    X, y = data.load_train()
    features = data.feature_columns(X)
    y = np.asarray(y).flatten()

    # Extract defects only
    defect_mask = y == 1
    X_defects = X[defect_mask][features].copy()
    defect_indices = np.where(defect_mask)[0]

    print(f"\nDefect samples: {len(X_defects)}")
    print(f"Features: {len(features)}")

    # Impute
    X_defects_imp = X_defects.fillna(X_defects.median())

    # Standardize
    scaler = StandardScaler()
    X_defects_scaled = scaler.fit_transform(X_defects_imp)

    print("\n" + "=" * 80)
    print("PART 1: DIMENSIONALITY REDUCTION")
    print("=" * 80)

    # PCA
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(X_defects_scaled)
    print(f"\nPCA variance explained (PC1, PC2): {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total variance (first 2 PCs): {sum(pca.explained_variance_ratio_):.3f}")

    # Try UMAP if available
    try:
        from umap import UMAP
        umap = UMAP(n_components=2, random_state=42, n_neighbors=5)
        umap_2d = umap.fit_transform(X_defects_scaled)
        has_umap = True
        print("UMAP projection computed")
    except ImportError:
        has_umap = False
        print("UMAP not available (optional)")

    # --- PART 2: CLUSTERING ---
    print("\n" + "=" * 80)
    print("PART 2: KMEANS CLUSTERING")
    print("=" * 80)

    cluster_results = {}

    for k in range(2, 5):
        print(f"\nK={k}:")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_defects_scaled)

        silhouette_avg = silhouette_score(X_defects_scaled, cluster_labels)
        cluster_results[k] = {
            "kmeans": kmeans,
            "labels": cluster_labels,
            "silhouette": silhouette_avg,
            "centers": kmeans.cluster_centers_,
        }

        # Cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"  Silhouette score: {silhouette_avg:.4f}")
        print(f"  Cluster sizes: {dict(zip(unique, counts))}")

        # Intra-cluster distance (cohesion)
        for c in unique:
            c_samples = X_defects_scaled[cluster_labels == c]
            if len(c_samples) > 1:
                intra_dist = pdist(c_samples).mean()
                print(f"    Cluster {c}: size={len(c_samples)}, avg intra-distance={intra_dist:.4f}")

    # Best clustering
    best_k = max(cluster_results, key=lambda k: cluster_results[k]["silhouette"])
    print(f"\nBest silhouette score: k={best_k} ({cluster_results[best_k]['silhouette']:.4f})")

    # --- PART 3: PROFILE BEST CLUSTERING ---
    print("\n" + "=" * 80)
    print(f"PART 3: DEFECT SUBTYPE PROFILES (K={best_k})")
    print("=" * 80)

    kmeans_best = cluster_results[best_k]["kmeans"]
    labels_best = cluster_results[best_k]["labels"]

    # Save cluster assignments
    cluster_assignment_df = pd.DataFrame({
        "defect_index": defect_indices,
        "cluster": labels_best,
    })

    for cluster_id in range(best_k):
        cluster_mask = labels_best == cluster_id
        cluster_size = cluster_mask.sum()
        X_cluster = X_defects_imp[cluster_mask]

        print(f"\nCluster {cluster_id} (n={cluster_size}):")

        # Cluster feature profile
        cluster_mean = X_cluster.mean()
        cluster_std = X_cluster.std()

        # Compute difference from global defect mean
        global_mean = X_defects_imp.mean()
        diff = cluster_mean - global_mean

        # Top differentiating features
        top_feat_idx = np.argsort(np.abs(diff))[-5:][::-1]

        print(f"  Top 5 distinguishing features:")
        for rank, feat_idx in enumerate(top_feat_idx, 1):
            feat = features[feat_idx]
            feat_diff = diff.iloc[feat_idx]
            global_val = global_mean.iloc[feat_idx]
            cluster_val = cluster_mean.iloc[feat_idx]
            print(f"    {rank}. {feat}: cluster={cluster_val:.4f} vs global={global_val:.4f} (Δ={feat_diff:+.4f})")

        # High variance features in this cluster
        high_var_idx = np.argsort(cluster_std)[-3:][::-1]
        print(f"  Most variable features in this cluster:")
        for rank, feat_idx in enumerate(high_var_idx, 1):
            feat = features[feat_idx]
            cluster_var = cluster_std.iloc[feat_idx]
            global_var = X_defects_imp.std().iloc[feat_idx]
            print(f"    {rank}. {feat}: cluster_std={cluster_var:.4f} vs global_std={global_var:.4f}")

    # --- PART 4: FALSE NEGATIVE CLUSTER MEMBERSHIP ---
    print("\n" + "=" * 80)
    print("PART 4: FALSE NEGATIVE CLUSTER MEMBERSHIP")
    print("=" * 80)

    # Load FN indices from OOF
    oof_df = pd.read_csv(config.ARTIFACTS / "oof_predictions.csv")
    oof_proba = oof_df["oof_proba"].values
    threshold = 0.00583
    oof_pred = (oof_proba >= threshold).astype(int)

    fn_mask = (y == 1) & (oof_pred == 0)
    fn_indices_global = np.where(fn_mask)[0]

    if len(fn_indices_global) > 0:
        print(f"\nFalse negative samples: {len(fn_indices_global)}")

        for fn_idx in fn_indices_global:
            # Map to defect index
            defect_pos = np.where(defect_indices == fn_idx)[0]
            if len(defect_pos) > 0:
                cluster_id = labels_best[defect_pos[0]]
                prob = oof_proba[fn_idx]
                print(f"  FN sample {fn_idx}: cluster {cluster_id}, prob={prob:.6f}")

    # Save results
    cluster_assignment_df.to_csv(config.ARTIFACTS / "defect_cluster_assignments.csv", index=False)
    pd.DataFrame(X_defects_scaled).to_csv(config.ARTIFACTS / "defect_scaled_features.csv", index=False)
    pd.DataFrame(pca_2d, columns=["PC1", "PC2"]).to_csv(config.ARTIFACTS / "defect_pca_2d.csv", index=False)

    if has_umap:
        pd.DataFrame(umap_2d, columns=["UMAP1", "UMAP2"]).to_csv(config.ARTIFACTS / "defect_umap_2d.csv", index=False)

    print(f"\nSaved cluster assignments → defect_cluster_assignments.csv")

    # --- VISUALIZATION (simple text-based) ---
    print("\n" + "=" * 80)
    print("PART 5: VISUALIZATION SUMMARY")
    print("=" * 80)

    print("\nCluster distribution in 2D PCA space:")
    print("(Visual representation would require matplotlib)")
    print(f"  Cluster 0 center (PCA): ({pca_2d[labels_best == 0, 0].mean():.2f}, {pca_2d[labels_best == 0, 1].mean():.2f})")
    if best_k > 1:
        print(f"  Cluster 1 center (PCA): ({pca_2d[labels_best == 1, 0].mean():.2f}, {pca_2d[labels_best == 1, 1].mean():.2f})")
    if best_k > 2:
        print(f"  Cluster 2 center (PCA): ({pca_2d[labels_best == 2, 0].mean():.2f}, {pca_2d[labels_best == 2, 1].mean():.2f})")

    return cluster_assignment_df, labels_best, X_defects_scaled, pca_2d


def main():
    cluster_assignment_df, labels_best, X_defects_scaled, pca_2d = analyze_positive_clustering()

    # --- FINAL ANALYSIS ---
    print("\n" + "=" * 80)
    print("SUMMARY & QUESTION")
    print("=" * 80)

    unique_clusters, counts = np.unique(labels_best, return_counts=True)
    print(f"\nDefect clustering results:")
    print(f"  Number of clusters: {len(unique_clusters)}")
    print(f"  Cluster sizes: {dict(zip(unique_clusters, counts))}")

    print("\nFinal Question: Do multiple hidden defect mechanisms exist?")
    print("\nAnalysis:")

    if len(unique_clusters) > 1:
        min_size = counts.min()
        max_size = counts.max()

        if min_size > 5:  # Clear, balanced clusters
            print(f"  ✓ YES - Found {len(unique_clusters)} distinct defect subtypes")
            print(f"  Cluster balance: {min_size}-{max_size} samples (reasonable separation)")
            print(f"  → PROCEED WITH TASK 3 (subtype distance feature engineering)")
        elif len(unique_clusters) == 2 and min_size >= 2:
            print(f"  ⚠ MAYBE - Found {len(unique_clusters)} clusters but small minority")
            print(f"  Cluster sizes: {counts}")
            print(f"  → CONSIDER TASK 3 (subtype distance features)")
        else:
            print(f"  ✗ NO - Clusters are too imbalanced or small")
            print(f"  → SKIP TASK 3 (not enough structure)")
    else:
        print(f"  ✗ NO - Only 1 cluster found")
        print(f"  → SKIP TASK 3 (homogeneous defect population)")

    print(f"\nTask 2 complete.")


if __name__ == "__main__":
    main()
