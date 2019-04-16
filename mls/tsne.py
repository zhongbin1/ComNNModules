from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_reduce(vectors, labels, num_components=2, perplexity=30, num_iter=1000, learning_rate=100.0, init="pca",
                metric="euclidean", verbose=2, plot=True, show_label=False, show=False, save_name="tsne",
                save_format="pdf"):
    """TSNE
    cf. https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    :param vectors: input features, ndarray.
    :param labels: input labels, list.
    :param num_components: dimension of the embedded space.
    :param perplexity: the perplexity is related to the number of nearest neighbors that is used in other
                       manifold learning algorithms, range [5, 50].
    :param num_iter: maximum number of iterations for the optimization. Should be at least 250.
    :param learning_rate: the learning rate for t-SNE is usually in the range [10.0, 1000.0].
    :param metric: use when calculating distance between instances in a feature array.
    :param init: initialization of embedding.
    :param verbose: verbosity mode.
    :param plot: if visualize.
    :param show_label: plot labels.
    :param show: if show the graph in a window.
    :param save_name: save name.
    :param save_format: save format.
    :return: reduced vectors.
    """
    tsne = TSNE(perplexity=perplexity,
                n_components=num_components,
                init=init,
                n_iter=num_iter,
                learning_rate=learning_rate,
                metric=metric,
                verbose=verbose)
    new_vectors = tsne.fit_transform(vectors)

    if plot:
        plt.figure(figsize=(10, 10))
        x_feature = new_vectors[:, 0]
        y_feature = new_vectors[:, 1]
        plt.scatter(x_feature, y_feature, c="r", marker="o", label=labels if show_label else "", alpha=0.8)
        plt.savefig(".".join([save_name, save_format]), format=save_format)
        if show:
            plt.show()

    return new_vectors
