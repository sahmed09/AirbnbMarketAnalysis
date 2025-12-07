# ==============================
# K-Means Clustering for NYC Airbnb Neighborhoods
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ------------------------------------
# Load & Prepare Dataset
# ------------------------------------
df = pd.read_csv('Dataset/AB_NYC_2019.csv')
df["last_review"] = pd.to_datetime(df["last_review"])
df["reviews_per_month"] = df["reviews_per_month"].fillna(df["reviews_per_month"].mean())
df.drop(columns=["host_name", "last_review"], inplace=True)

# Keep only valid listings
df = df[df["price"] > 0]

# Compute estimated occupancy rate from availability_365
df["estimated_occupancy_rate"] = (1 - df["availability_365"] / 365).clip(0, 1)

# ------------------------------------
# Aggregate features at neighborhood level
# ------------------------------------
neigh_df = df.groupby("neighbourhood").agg(
    mean_price=("price", "mean"),
    mean_occupancy_rate=("estimated_occupancy_rate", "mean")
).reset_index()

print("\nNeighborhood aggregated dataset:")
print(neigh_df.head())

# ------------------------------------
# Select Features for Clustering
# ------------------------------------
features = ["mean_price", "mean_occupancy_rate"]
X = neigh_df[features].values

# Scale values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------
# PCA for Visualization Only
# ------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Projection (Unclustered Neighborhoods)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("neighborhood_PCA_unclustered.png", dpi=1200)
plt.show()

# ====================================================
# K-MEANS CLUSTERING
# ====================================================

silhouette_scores = []
db_scores = []
K_values = range(2, 8)

print("\n=== KMeans Clustering Evaluation ===")
for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)

    silhouette_scores.append(sil)
    db_scores.append(dbi)

    print(f"K={k} | Silhouette Score={sil:.4f} | Davies–Bouldin Index={dbi:.4f}")

# Choose best K using the highest silhouette score
best_k = K_values[np.argmax(silhouette_scores)]
print(f"\nBest K (based on Silhouette Score): {best_k}")

# Train final clustering model
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
neigh_df["cluster"] = kmeans_final.fit_predict(X_scaled)

# Final evaluation metrics
final_sil = silhouette_score(X_scaled, neigh_df["cluster"])
final_db = davies_bouldin_score(X_scaled, neigh_df["cluster"])

print("\nFinal KMeans Model Performance:")
print(f"Silhouette Score: {final_sil:.4f}")
print(f"Davies–Bouldin Index: {final_db:.4f}")

# ====================================================
# VISUALIZATION
# ====================================================

# Cluster Visualization in PCA space
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=neigh_df["cluster"], palette="Set2", s=80)
plt.title(f"KMeans Clusters (K={best_k}) Based on Avg Price & Occupancy Rate")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.savefig("KMeans_clusters_PCA.png", dpi=1200)
plt.show()

# Center values (inverse transformed)
centers_scaled = kmeans_final.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

cluster_centers_df = pd.DataFrame(centers_original, columns=features)
cluster_centers_df["cluster"] = range(best_k)
cluster_centers_df.set_index("cluster", inplace=True)

print("\nCluster Centers (Original Scale):")
print(cluster_centers_df)

# Heatmap visualization of cluster centers
plt.figure(figsize=(10, 5))
sns.heatmap(cluster_centers_df, annot=True, cmap="viridis", fmt=".2f")
plt.title("Cluster Profile: Avg Price vs Occupancy Rate")
plt.savefig("cluster_profile_heatmap.png", dpi=1200)
plt.show()

cluster_counts = neigh_df['cluster'].value_counts().sort_index()

plt.figure(figsize=(6, 5))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Set2")

plt.title("Number of Neighborhoods in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Neighborhoods")
plt.xticks(cluster_counts.index)

# Add text labels above bars
for index, value in enumerate(cluster_counts.values):
    plt.text(index, value + 1, str(value), ha='center', fontsize=12)

plt.tight_layout()
plt.savefig("neighborhood_cluster_distribution.png", dpi=1200)
plt.show()

print("\nClustering Completed.")
