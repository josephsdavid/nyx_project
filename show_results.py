import numpy as np
import umap
import pickle

viz = np.load("viz.npy")
recs = np.load("recs.npy")
inputs = np.load("input.npy")
mov_map=pickle.load(open("movie_map.pkl", 'rb'))

umap_dim=2
umap_neighbors = 10
umap_min_distance=0.
umap_metric="euclidean"

manifold = umap.UMAP(
        metric=umap_metric,
        n_components=umap_dim,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_distance
        )

embedding = manifold.fit_transform(viz[:1000, :])

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.scatter(
        embedding[:,0],
        embedding[:,1]
        )

plt.savefig("fig_test.png")
