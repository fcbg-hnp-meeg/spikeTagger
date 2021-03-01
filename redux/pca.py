# pca.py
from sklearn.decomposition import PCA

def pca(X, n_classes, y=None):
    """

    Args:
        X:
        y:

    Returns:

    """
    print('------------------PCA------------')
    pca = PCA(n_components=0.95, svd_solver='full')
    # pca = PCA(n_components='mle')
    pca.fit(X)
    return (pca.transform(X), pca.components_.shape[0], pca)

