from sklearn import cluster, metrics


def kmeans_clustering(vectors, num_clusters, init="k-means++", n_init=20, max_iter=500, tol=1e-12, verbose=0):
    """K-means clustering
    cf. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    :param vectors: input features, ndarray.
    :param num_clusters: number of clusters to form as well as the number of centroids to generate.
    :param init: method for initialization, defaults to 'k-means++'.
    :param n_init: number of time the k-means algorithm will be run with different centroid seeds.
    :param max_iter: maximum number of iterations of the k-means algorithm for a single run.
    :param tol: relative tolerance with regards to inertia to declare convergence.
    :param verbose: verbosity mode.
    :return: labels, centroids and scores.
    """
    kmeans = cluster.KMeans(n_clusters=num_clusters,
                            init=init,
                            n_init=n_init,
                            max_iter=max_iter,
                            tol=tol,
                            verbose=verbose,
                            precompute_distances="auto",
                            random_state=None,
                            n_jobs=10,
                            algorithm="auto")
    kmeans.fit(vectors)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = kmeans.score(vectors)
    silhouette_score = metrics.silhouette_score(vectors, labels, metric="cosine")
    return labels, centroids, score, silhouette_score
