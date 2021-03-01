# train_pipeline.py
from EEGEpilecticSpikeTagger.redux.ident import ident
from EEGEpilecticSpikeTagger.preprocessing.standardize_data import standardize_data

import numpy as np

from sklearn import svm
from sklearn import  metrics
from sklearn.ensemble import IsolationForest
from sklearn.cluster \
    import KMeans,\
    AffinityPropagation,\
    AgglomerativeClustering
from sklearn.mixture \
    import GaussianMixture,\
    BayesianGaussianMixture

from sklearn.model_selection\
    import StratifiedKFold



def clf_training(domain, n_kfold_split, n_features, features_set, labels_set, clf_type, n_classes,
                 additional_clf_params=None,
                 mode_training=True):
    """
    This function responsible for classifiers training.
    Classifiers are trained following k-fold procedure (defautl:=10 wraps)
    Additionally, it records classifiers: - settings
                                          - performances as: (balanced)accuracies,... and f1_scores
    Args:
        domain:
        n_kfold_split (int): number (k) of folds to be used
        n_features (int): number of features to be considered
        features_set (np.array<float>): shape(n_epochs,n_features)
        labels_set (np.array<int>): shape(n_epochs,)
        clf_type (string): specifies type of classifier to be use {'kmeans':,
                                                                   'svm': support vector machine,
                                                                   'if': isolation forest,
                                                                   'em': expectation maximization (gaussian model),
                                                                   'bgm': bayesian gaussian model}
        n_classes (int): number of classes/cluster/component whithin data
        additional_clf_params: Could be use

    Returns:
        A list of so called 'clf_info' entries:
        - clf_list (list<clf_info>):
            - clf_info (list):= [settings, kfd_accuracy_mean, kfd_f1_score_mean, accr_container, f1sc_container]
            _ settings (tuple):=(domain, clf, additional_clf_params, n_features)
                * domain (string): features domain {'t': time domain, 'ft': time-frequency domain}
                * kft_accuracy_mean (float): mean of accuracies upon k-folds
                * kfd_f1_score_mean (float): mean of f1-scores upon k-folds
                * accr_container (list<float>): actual accuracies computed upon the k-folds
                * f1sc_container (list<float>): actual f1-scores computed upon the k-folds


    """
    print('----------------- clf training  --------------------',domain, clf_type, additional_clf_params,'md.tr:',mode_training)  # , type(epks_set))
    # empty containers intitialization
    clf_list = []
    accr_container = []
    #f1sc_container = []
    #f1scw_container = []
    #f1scma_container = []
    #f1scmi_container = []
    #precision_container = []
    #cm_container = []
    #mcm_container = []

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
        print('lkajdflkj----- bad clf_type')
        clf=None

    # classifier's settings recording
    settings = (domain, clf, additional_clf_params, n_features)

    if mode_training:
        # cross-validation indexes initialization
        skf = StratifiedKFold(n_splits=n_kfold_split)
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
            accr_container.append(accrcy)
            #f1sc_container.append(f1_scr)
            #f1scw_container.append(f1_scrw)
            #f1scmi_container.append(f1_scrmi)
            #f1scma_container.append(f1_scrma)
            #precision_container.append(prcsn)
            #cm_container.append(conf_mat)
            #mcm_container.append(multlab_conf_mat)

        # accuracies/f1-score means computation over the k-folds
        kfd_accuracy_mean = np.mean(np.asarray(accr_container))
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
        print('slkdjf')
        clf.fit(features_set,labels_set)
        predicted = clf.predict(features_set)
        print(type(features_set),features_set.shape,predicted.shape,labels_set.shape)
        accuracy = metrics.balanced_accuracy_score(labels_set,
                                                 predicted)  # prob with 'if' need to average= specification
        clf_info = [settings,accuracy]



    clf_list.append(clf_info)

    return (clf_list)
def train_pipeline(epks_set, labels_set, clf_type, domain, n_classes, features_fun=lambda a: a,
                      ch_selection=slice(None),
                      standardization=True, redux_procedure=ident,
                      mode_training = True):
    """
    This is the main function of the program. Given a collection of epochs it will
        - data preprocessing: (when needed)
        - feature extraction:
        - feature values standardization:
        - dimension reduction:
        - classifiers and performance collection:
    Args:
        epks_set (mne.Epoch): A collection of mne.Epoch obj. on which the procedure is to be performed.
        labels_set (np.array<int>): A labels collection corresponding to the epks_set
        clf_type (str): A string to specify which classifier is to be used:
                        - 'kmeans': k-means algorithm
                        - 'svm':    Support Vector Machine
                        - 'if':     Isolation Forest
                        - 'em':     Expectation-Maximization
        features_fun (function): Specification of the procedure in charge of features extraction
                                 - defautl:=Identity(ident)
        ch_selection (slice¦tuple): Specification of the channel(s) of interest
                                    - default:=slice(None) = all channels
        standardization (bool): Specification whether data(features) are to be standardized or not
        redux_procedure (function): Specification of the reduction procedure to be applied on the data.
                                    - defautl:=Identity(ident)

    Returns (list<list<info_format>:
                                    info_format: list<>

    """
    print('\n----------------- Training Pipeline --------------------', type(epks_set),mode_training,labels_set.shape)
    print(labels_set)
    print('=================================')
    redux = None
    # feature extraction
    features = features_fun(epks_set)

    # relevant channel election
    # irrelevant channels elision
    data = features[:, ch_selection, :]
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])  # ,order='A')
    n_features = data.shape[-1]

    # features standardization #todo what if?? col.standdev==0
    if standardization:
        import preprocessing
        data = standardize_data(data)

    # (dimensional) reduction
    print(redux_procedure)
    #if redux_procedure != ident:
    data, n_features, redux = redux_procedure(data, n_classes, labels_set)
    # training
    clf = clf_training(domain, 10, n_features, data, labels_set, clf_type, n_classes,additional_clf_params=redux,mode_training=mode_training)

    return (clf)
