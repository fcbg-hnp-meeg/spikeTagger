# train_clf
# classifiers
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering  # No predict fun.. only fit_predict
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn import svm

# k-fold
from sklearn.model_selection import StratifiedKFold

import numpy as np
from sklearn import metrics

def train_clf(features_set,clf_type,n_classes,labels_set,k_folds,additional_params):
    """

    Args:
        features_set (list<np.array>): The features on witch the classifiers is to be train on
        clf_type (str): Specification of the type of classifier to use
        n_classes (int): Number of classes to discriminate
        labels_set (list<int>): Labels corresponding to the features samples
        k_folds (int): Number of folds to use when willing to perform k-folds procedure (default=0)
        additional_params: Additionnal relevant informations (Here: specification of the reduction procedure applied on
                           the features

    Returns (list<info>): A list of relevant informations.
                          list_element := list[settings,mean_of_metric_over_k_folds]
                          settings := tuple(classifier,reduction_procedure,nb_features)

    """
    redux_proc = additional_params[0]
    n_features = features_set.shape[-1]

    clf_list = []
    # classifier selection
    if clf_type == 'kmeans':
        clf = KMeans(n_clusters=n_classes)
    elif clf_type == 'svm':
        # SVM- support vector machine
        clf = svm.SVC()
    elif clf_type == 'hc':  # hierarchical clustering
        clf = AgglomerativeClustering(n_clusters=n_classes, affinity='euclidean', linkage='ward')
    elif clf_type == 'if':  # isolation forest
        clf = IsolationForest()
        labels_set[labels_set==0]=-1 # hack for 2Tags classification # todo what about 3tags?
    elif clf_type == 'em':
        if sum(labels_set==-1) !=0:
            labels_set[labels_set==-1]=0
        # n_components shall be chosen via bic criterion
        # cv_type: full(default)/spherical/tied/dag
        clf = GaussianMixture(n_components=n_classes, covariance_type='full')
    elif clf_type == 'ap':  # affinity propagation
        clf = AffinityPropagation(random_state=5, max_iter=1000)  # convergence issues might need tuning
    elif clf_type == 'bgm':  # BayesianGaussianMixture
        if sum(labels_set==-1) !=0:
            labels_set[labels_set==-1]=0
        clf = BayesianGaussianMixture(n_components=n_classes, max_iter=200)
    else:  # error handling (default behaviour) todo
        print('WARNING!!! ----- bad clf_type')
        clf=None

    # classifier's settings recording
    #settings = (domain, clf, additional_params, n_features)
    settings = (clf, redux_proc, n_features)

    # perfomance metric
    performances_container = []
    if k_folds !=0:
        print('performing (k)-',k_folds,'-folds')
        # cross-validation indexes initialization
        skf = StratifiedKFold(n_splits=k_folds)
        # classifiers training & perfomances recodings according to k-fold policy
        for tri, tei in skf.split(features_set, labels_set):
            # training examples selection
            ff_tr = features_set[tri, :]

            # classifier training/fit
            clf.fit(ff_tr, labels_set[tri])

            # testing examples selection
            ff_te = features_set[tei, :]

            # calssifier prediction  computation
            predicted = clf.predict(ff_te)

            # accuracy computation
            accrcy = metrics.balanced_accuracy_score(labels_set[tei],
                                                 predicted)  # prob with 'if' need to average= specification
            # other metrics
            # f1-score computation
            #f1_scr = metrics.f1_score(labels_set[tei], predicted,average=None)
            #f1_scrw = metrics.f1_score(labels_set[tei], predicted,average='weighted')
            #f1_scrmi = metrics.f1_score(labels_set[tei], predicted,average='micro')
            #f1_scrma = metrics.f1_score(labels_set[tei], predicted,average='macro')
            # precision computation
            #prcsn = metrics.precision_score(labels_set[tei],predicted,average='weighted')
            # confusion matrices
            #conf_mat = metrics.confusion_matrix(labels_set[tei],predicted)
            #multlab_conf_mat = metrics.multilabel_confusion_matrix(labels_set[tei],predicted)

            # accuracy correction
            #if (accrcy < .5):
            #    accrcy = 1 - accrcy
            #    # f1_scr = 1 - f1_scr

            # accuracies and f1-scores recording
            performances_container.append(accrcy)
            #f1sc_container.append(f1_scr)
            #f1scw_container.append(f1_scrw)
            #f1scmi_container.append(f1_scrmi)
            #f1scma_container.append(f1_scrma)
            #precision_container.append(prcsn)
            #cm_container.append(conf_mat)
            #mcm_container.append(multlab_conf_mat)

        # accuracies/f1-score means computation over the k-folds
        kfd_accuracy_mean = np.mean(np.asarray(performances_container))
        #kfd_f1_score_mean = np.mean(np.asarray(f1sc_container)) # !! dim  prob -> verif #class within sample
        #kfd_f1_scorew_mean = np.mean(np.asarray(f1scw_container))
        #kfd_f1_scoremi_mean = np.mean(np.asarray(f1scmi_container))
        #fd_f1_scorema_mean = np.mean(np.asarray(f1scma_container))
        #kfd_prcsn_mean = np.mean(np.asarray(precision_container))

        # records formatting
        clf_info = [settings,
                    kfd_accuracy_mean]#,
                    #kfd_f1_score_mean,
                    #kfdf1_scorew_mean, kfd_f1_scoremi_mean, kfd_f1_scorema_mean,
                    #f1sc_container]#,
                    #f1scw_container, f1scma_container, f1scma_container]
    else:
        print('slkdjf',features_set.shape,labels_set.shape)
        clf.fit(features_set,labels_set)
        predicted = clf.predict(features_set)
        print('=?',type(features_set),features_set.shape,predicted.shape,labels_set.shape)
        accuracy = metrics.balanced_accuracy_score(labels_set,
                                                 predicted)  # prob with 'if' need to average= specification
        #accuracy = 0
        clf_info = [settings,accuracy]



    clf_list.append(clf_info)

    return (clf_list)
