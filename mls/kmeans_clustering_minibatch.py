from sklearn import cluster, metrics


def kmeans_clustering(vectors, num_clusters, init="k-means++", n_init=3, max_iter=500, batch_size=64, tol=1e-3,
                      verbose=True):
    """Minibatch kmeans clustering
    cf. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
    :param vectors: input features, ndarray.
    :param num_clusters: number of clusters to form as well as the number of centroids to generate.
    :param init: method for initialization, defaults to 'k-means++'.
    :param n_init: number of random initializations that are tried.
    :param max_iter: maximum number of iterations over the complete dataset before stopping.
    :param batch_size: size of the mini batches.
    :param tol: control early stopping based on the relative center changes.
    :param verbose: verbosity mode.
    :return:
    """
    kmeans = cluster.MiniBatchKMeans(n_clusters=num_clusters,
                                     init=init,
                                     max_iter=max_iter,
                                     n_init=n_init,
                                     batch_size=batch_size,
                                     tol=tol,
                                     verbose=verbose,
                                     max_no_improvement=10,
                                     reassignment_ratio=0.01,
                                     random_state=None)
    kmeans.fit(vectors)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = kmeans.score(vectors)
    silhouette_score = metrics.silhouette_score(vectors, labels, metric="cosine")
    return labels, centroids, score, silhouette_score
