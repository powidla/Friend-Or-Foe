import os
import json

#stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from models import DataLoader, FairSC, FairDEN
from metrics import dcsi_score


def run_clustering_comparison(X_all, sensitive_attributes=None, n_clusters=3):
    """Compare clustering methods"""

    X_scaled = StandardScaler().fit_transform(X_all)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

   # Baselines
    methods = {
        "DBSCAN": AgglomerativeClustering(n_clusters=n_clusters),
        "K-Means": KMeans(n_clusters=n_clusters, random_state=42),
    }

    # FairDen + FairSC
    data_loader = DataLoader(X_scaled, sensitive_attributes, n_clusters)

    try:
        methods["FairSC"] = FairSC(data_loader).run
    except Exception as e:
        print(f"FairSC init failed: {str(e)}")

    try:
        methods["FairDEN"] = FairDEN(data_loader, min_pts=5).run
    except Exception as e:
        print(f"FairDEN init failed: {str(e)}")

    grid_methods = [
        ["DBSCAN", "K-Means"],
        ["FairSC", "FairDEN"]
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    results = []

    for i in range(2):
        for j in range(2):
            name = grid_methods[i][j]
            if name not in methods:
                axs[i,j].axis('off')
                continue

            try:
                model = methods[name]
                labels = model(n_clusters) if callable(model) else model.fit_predict(X_scaled)

                unique_labels = set(labels)
                if len(unique_labels) > 1:
                    results.append({
                        "Method": name,
                        "Silhouette": silhouette_score(X_scaled, labels), # SC
                        "DCSI": dcsi_score(X_scaled, labels),  #  DCSI
                        "Clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
                        # "Noise Points": np.sum(labels == -1) if -1 in labels else 0
                    })
                    scatter = axs[i,j].scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                                              cmap='coolwarm', alpha=0.4, s=20)
                    axs[i,j].set_title(f"{name}", pad=10)
                    axs[i,j].set_xlabel("PC1")
                    axs[i,j].set_ylabel("PC2")
                    axs[i,j].grid(True, linestyle='--', alpha=0.5)

                    plt.colorbar(scatter, ax=axs[i,j], label="Cluster")

            except Exception as e:
                print(f"{name} failed: {str(e)}")
                axs[i,j].axis('off')

    results_df = pd.DataFrame(results)
    print("\nClustering Performance Comparison:")
    print(results_df.to_markdown(index=False, floatfmt=".3f"))

    results_json = results_df.set_index("Method")[["Silhouette", "DCSI"]].to_dict(orient="index")

    with open("results.json", "w") as f:
         json.dump(results_json, f, indent=4)

    return results_json

X = np.load("FOFdata/Clustering/AGORA/100/np/AG_US-I-100.npy")

# Run
results = run_clustering_comparison(
X,
sensitive_attributes=None,
n_clusters=3
)

