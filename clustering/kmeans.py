
import math
import argparse
import pandas
import numpy as np

from time import time
from pandas.core.frame import DataFrame
from pandas.core.series import Series

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

IGNORED_COLUMNS = [
    'cluster',
    'quality',
    'class'
]


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


    # Remove columns that we don't cluster on
    # This is built for the wine dataset and the TwoDimHard.
    # It'd be nice if this was more intelligent, but alas.
    for column in IGNORED_COLUMNS:
        if column in df:
            df = df.drop(column, 1)

    # Normalize non-ID columns
    ids = []
    for column in df.columns:
        if column != 'ID':
            ids.append(column)

    df[ids] = df[ids].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df


def vectorize(row: Series) -> tuple:
    """Data-specific vectorization"""
    l = row.tolist()

    # Grab everything except ID (first column)
    # and _cluster/_distance (last 2 columns)
    return np.array(l[1:-2])


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
    print('ID\t\tCluster')
    f = open('out.csv', 'w')

    for i, row in df.iterrows():
        print('{:.0f}\t\t{:.0f}'.format(row['ID'], row['_cluster']))
        f.write('{:.0f},{:.0f}\n'.format(row['ID'], row['_cluster']))

    f.close()


def print_statistics(df: DataFrame, centroids: list):
    """print SSE and SSB statistics"""
    total_sse = 0
    ssb = 0

    cluster_ssb = []

    print('---')
    print('Cluster\t\tSSE')
    for cluster, centroid in enumerate(centroids):
        sse = 0
        ssb = 0
        subset = df.loc[df['_cluster'] == cluster]

        # Calculate SSB between this cluster and all others
        cluster_size = len(subset)

        for cl2, c2 in enumerate(centroids):
            if cl2 != cluster:
                ssb = ssb + cluster_size * (distance(centroid.position, c2.position) ** 2)

        cluster_ssb.append(ssb)

        # Calculate SSE between points and the cluster mean
        for i, row in subset.iterrows():
            sse = sse + row['_distance'] ** 2

        total_sse = total_sse + sse
        print('{}\t\t{}'.format(cluster, sse))

    print('Total SSE\t{}'.format(total_sse))
    print('---')
    print('Cluster\t\tSSB')
    for cluster, ssb in enumerate(cluster_ssb):
        print('{}\t\t{}'.format(cluster, ssb))


def print_cluster_scatterplot(df: DataFrame, centroids: list):
    """Only works for TwoDimHard dataset"""

    # Cluster palette
    colors = [
        'green',
        'orange',
        'blue',
        'purple',
        'tan',
        'yellowgreen',
        'royalblue',
        'mediumvioletred',
        'pink',
        'salmon'
    ]

    # x = []
    # y = []
    # for i, centroid in enumerate(centroids):
    #     # Centroid position
    #     x.append(centroid.position[0])
    #     y.append(centroid.position[1])

    #     # Plot points in the centroid
    #     subset = df.loc[df['_cluster'] == i]
    #     plt.scatter(subset['X.1'], subset['X.2'], c=colors[i], s=5)

    # add + markers for all centroids
    # plt.scatter(x, y, c='red', marker='+', s=50)

    y = df['_cluster']
    x = df.drop(['ID', '_cluster', '_distance'], axis=1)
    x_norm = (x - x.min()) / (x.max() - x.min())

    pca = PCA(n_components=2)
    transformed = DataFrame(pca.fit_transform(x_norm))

    # lda = LDA(n_components=2)
    # transformed = DataFrame(lda.fit_transform(x_norm, y))

    for i, centroid in enumerate(centroids):
        plt.scatter(
            transformed[y==i][0],
            transformed[y==i][1],
            label='Class ' + str(i),
            c = colors[i],
            s = 5
        )

    plt.legend()
    plt.show()


def k_means_clustering(df: DataFrame, k: int, statistics: bool) -> None:
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

    if statistics:
        print_statistics(df, centroids)
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

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show SSE/SSB and PCA scatterplots'
    )

    parser.add_argument('filename')

    args = parser.parse_args()
    df = pandas.read_csv(args.filename)

    df = preprocess(df)

    k_means_clustering(df, args.k, args.stats)
