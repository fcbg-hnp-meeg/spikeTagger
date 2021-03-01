# ica.py
from sklearn.decomposition import FastICA
import numpy as np

def ica(X, n_classes, y=None):
    """

    Args:
        X:
        y:

    Returns:

    """
    print('------------------ICA------------',X.shape)
    #ica = FastICA(n_components=n_classes)
    #ica.fit(X)

    covvar_mat = np.corrcoef(X,rowvar=False)
    rank = np.linalg.matrix_rank(covvar_mat)
    print('rank---------------:',rank)
    ica = FastICA(n_components=rank,max_iter=1000)
    ica.fit(X)
    return (ica.transform(X), ica.n_components, ica)
    #return(rank)
    #return (ica.transform(X), ica.n_components, ica)