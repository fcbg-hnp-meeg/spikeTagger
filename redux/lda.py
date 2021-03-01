# lda.py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda(X, n_classes, y):
    print('------------------LDA------------')
    lda = LinearDiscriminantAnalysis()#n_components=min(n_classes - 1,X.shape[-1]))  # =min(n_classes-1,X.shape[-1]))
    lda.fit(X, y)
    print('---------->>>>>>>>>>>>>>>>>', lda.classes_)
    return (lda.transform(X), lda.n_components, lda)
