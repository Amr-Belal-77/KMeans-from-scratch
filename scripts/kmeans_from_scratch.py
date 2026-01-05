"""
K-Means Clustering From Scratch (NumPy implementation)
Author: Amr Belal
Description:
    Simple and reproducible implementation of K-Means without using scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import argparse
from pathlib import Path


def initialize_centroids(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Randomly select k samples as initial centroids."""
    np.random.seed(seed)
    random_idx = np.random.choice(len(X), size=k, replace=False)
    return X[random_idx]


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest centroid."""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recalculate centroids as mean of points in each cluster."""
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def has_converged(old: np.ndarray, new: np.ndarray, tol: float = 1e-4) -> bool:
    """Check if centroids stopped moving significantly."""
    return np.all(np.linalg.norm(new - old, axis=1) < tol)


def kmeans(X: np.ndarray, k: int = 3, max_iters: int = 100, tol: float = 1e-4, seed: int = 42):
    """Full K-Means clustering algorithm."""
    centroids = initialize_centroids(X, k, seed)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if has_converged(centroids, new_centroids, tol):
            break
        centroids = new_centroids
    return centroids, labels


def visualize_clusters(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray, save_path: Path = None):
    """Plot clusters and centroids."""
    plt.figure(figsize=(7, 5))
    for i in np.unique(labels):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i}")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", s=200, marker="X", label="Centroids")
    plt.title("K-Means Clustering (from scratch)")
    plt.legend()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run K-Means clustering from scratch.")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    parser.add_argument("--samples", type=int, default=300, help="Number of sample points")
    parser.add_argument("--features", type=int, default=2, help="Number of features")
    parser.add_argument("--std", type=float, default=0.8, help="Cluster standard deviation")
    parser.add_argument("--output", type=str, default="assets/kmeans_result.png", help="Output plot path")
    args = parser.parse_args()

    X, _ = make_blobs(n_samples=args.samples, n_features=args.features, centers=args.k, cluster_std=args.std, random_state=42)

    centroids, labels = kmeans(X, k=args.k)
    visualize_clusters(X, centroids, labels, Path(args.output))

    print(f"✅ K-Means completed with {args.k} clusters.")
    print(f"Centroids:\n{centroids}")


if __name__ == "__main__":
    main()
