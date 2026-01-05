import numpy as np
from scripts.kmeans_from_scratch import kmeans

def test_shape():
    X = np.random.rand(10, 2)
    centroids, labels = kmeans(X, k=2)
    assert centroids.shape == (2, 2)
    assert len(labels) == len(X)
