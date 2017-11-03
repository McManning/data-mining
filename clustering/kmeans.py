
import math
import argparse
import pandas
import numpy as np

from time import time
from pandas.core.frame import DataFrame
from pandas.core.series import Series

import matplotlib.pyplot as plt


class Centroid:
    def __init__(self, position):
        """
            position: np.array
        """
        self.position = position

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.position, other.position)

        return False

    def __ne__(self, other):
        return not self.__eq__(other)


def preprocess(df: DataFrame) -> None:
    """Add extra attributes for tracking clusters and distances"""
    zeroes = np.zeros(len(df))

    df = df.assign(_cluster = Series(zeroes))
    df = df.assign(_distance = Series(zeroes))

    return df


def vectorize(row: Series) -> tuple:
    """Data-specific vectorization"""
    return np.array((row['X.1'], row['X.2']))


def distance(left, right) -> float:
    """Euclidean distance of two np.arrays"""
    return np.sqrt(np.sum((left - right) ** 2))


def set_initial_centroids(df: DataFrame, k: int) -> list:
    """ Pick K random entries to be centroids """
    samples = df.sample(k)
    return [Centroid(vectorize(sample)) for i, sample in samples.iterrows()]


def recompute_centroids(df: DataFrame, centroids: list)-> list:
    """
        Update centroids with new positions
    """
    # For each centroid, update its location to the mean of
    # all vectorized rows that are in its cluster
    new_centroids = []
    for i, centroid in enumerate(centroids):
        subset = df.loc[df['_cluster'] == i]

        new_centroids.append(Centroid(
            np.mean(
                np.array([
                    vectorize(row) for i, row in subset.iterrows()
                ]),
                axis = 0
            )
        ))

        print(i, new_centroids[i].position)

    return new_centroids


def assign_clusters(df: DataFrame, centroids: list) -> None:
    """Assign each row to its closest cluster"""

    for i, row in df.iterrows():
        d = None
        for cluster, centroid in enumerate(centroids):
            kd = distance(centroid.position, vectorize(row))
            if d == None or kd < d:
                d = kd
                closest = cluster

        df.set_value(i, '_cluster', closest)
        df.set_value(i, '_distance', d)


def same_centroids(start: list, end: list) -> bool:
    """check if the two centroid lists are equivalent"""
    for i in range(len(start)):
        if start[i] != end[i]:
            return False

    return True


def print_clusters(df: DataFrame):
    print(df)
    return

    for i, row in df.iterrows():
        print(row['ID'], row['_cluster'])


def print_cluster_scatterplot(df: DataFrame, centroids: list):
    # Cluster palette
    colors = ['green', 'orange', 'blue', 'purple']

    x = []
    y = []
    for i, centroid in enumerate(centroids):
        # Centroid position
        x.append(centroid.position[0])
        y.append(centroid.position[1])

        # Plot points in the centroid
        subset = df.loc[df['cluster'] == i+1]
        plt.scatter(subset['X.1'], subset['X.2'], c=colors[i], s=5)

    # add + markers for all centroids
    plt.scatter(x, y, c='red', marker='+', s=50)

    plt.show()



def k_means_clustering(df: DataFrame, k: int) -> None:
    """apply k-means and spit out some shenanigans"""

    """
        select K random points for initial centroids
        repeat
            form K clusters by assigning all points to their
            closest centroid (via euclidean distance)
            recompute centroid for each cluster
        until - the centroids don't change

        spit out a table mapping IDs to clusters
        spit out per-cluster SSE and overall SSE
        spit out SSB
    """

    centroids = set_initial_centroids(df, k)
    while True:
        assign_clusters(df, centroids)

        new_centroids = recompute_centroids(df, centroids)

        if same_centroids(centroids, new_centroids):
            break
        else:
            centroids = new_centroids

    print_clusters(df)

    print_cluster_scatterplot(df, centroids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-means clustering'
    )

    parser.add_argument(
        '--k',
        default=4,
        type=int,
        help='K-value'
    )

    parser.add_argument('filename')

    args = parser.parse_args()
    df = pandas.read_csv(args.filename)

    df = preprocess(df)

    k_means_clustering(df, args.k)
