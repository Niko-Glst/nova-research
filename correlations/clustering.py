"""Hierarchical clustering of assets by correlation, written in numpy.

Assets are merged bottom-up by the correlation distance

    d_ij = sqrt((1 - rho_ij) / 2)

which is a proper metric (Mantegna 1999; Lopez de Prado 2016): 0 for perfectly
correlated assets, 1/sqrt(2) for uncorrelated, 1 for perfectly anti-correlated.
Clusters are joined by average linkage, the mean distance between their
members. Single linkage chains everything into one long cluster on financial
data, and complete linkage is driven by the single most distant pair; average
linkage is the usual compromise.

The output uses scipy's linkage-matrix layout (each row: the two clusters
merged, their distance, the new cluster's size), so it is familiar to anyone
who has used scipy, without depending on it. The leaf order puts similar
assets next to each other, which is what makes a heatmap readable.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Clustering:
    linkage: np.ndarray  # (N - 1) x 4: [cluster_a, cluster_b, distance, size]
    order: tuple[int, ...]  # leaf order for display
    labels: tuple[str, ...]

    @property
    def ordered_labels(self) -> tuple[str, ...]:
        return tuple(self.labels[i] for i in self.order)


def correlation_distance(correlation: np.ndarray) -> np.ndarray:
    corr = np.clip(np.asarray(correlation, dtype=float), -1.0, 1.0)
    distance = np.sqrt((1.0 - corr) / 2.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def average_linkage(distance: np.ndarray) -> np.ndarray:
    """Agglomerative clustering with average linkage. O(N^3), fine for N < 200."""
    n = distance.shape[0]
    if n < 2:
        return np.empty((0, 4))
    # Active clusters: id -> member list. Ids 0..n-1 are leaves, n.. are merges.
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    rows = []
    next_id = n
    while len(members) > 1:
        ids = sorted(members)
        best = (np.inf, -1, -1)
        for ai, a in enumerate(ids):
            for b in ids[ai + 1 :]:
                d = float(distance[np.ix_(members[a], members[b])].mean())
                if d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        merged = members.pop(a) + members.pop(b)
        members[next_id] = merged
        rows.append([a, b, d, len(merged)])
        next_id += 1
    return np.array(rows, dtype=float)


def leaf_order(linkage: np.ndarray, n: int) -> tuple[int, ...]:
    """Left-to-right leaf order of the dendrogram."""
    if n == 1:
        return (0,)
    children = {n + i: (int(row[0]), int(row[1])) for i, row in enumerate(linkage)}

    def walk(node: int) -> list[int]:
        if node < n:
            return [node]
        left, right = children[node]
        return walk(left) + walk(right)

    return tuple(walk(n + len(linkage) - 1))


def cluster(correlation: np.ndarray, labels: list[str] | tuple[str, ...]) -> Clustering:
    if np.isnan(np.asarray(correlation, dtype=float)).any():
        raise ValueError("Cannot cluster a correlation matrix containing NaN.")
    linkage = average_linkage(correlation_distance(correlation))
    return Clustering(linkage, leaf_order(linkage, len(labels)), tuple(labels))
