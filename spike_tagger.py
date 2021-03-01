# testing_pipeline
import numpy as np
from configparser import ConfigParser
from pathlib import Path

from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering  # No predict fun.. only fit_predict
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn import svm
from misc import load_raws_within_dir
from misc import pklload, pkldump
from preprocessing import windower
from features.TimeDomain import  extract_td_features_from_epks
from features.FrequencyDomain import svm_features
from util import reshape_numpy
from preprocessing import standardize_data
from processing import band_filter
from processing import add_events_to_raws
from redux import ident,pca,ica,lda

import processing.add_events_to_raws as aetr
def main(data_directory_path, init_file):
    pred = pklload('predidcted.txt')
    wins = pklload('windows.txt')
    raws = pklload('r_data.txt')

    print('DATA_DIRECTORY:{}'.format(data_directory_path))
    print('CONFIGURATION_FILE: {}'.format(init_file))

    # settings recovering
    # via settings.ini file
    print('Parameters recovering..')
    config = ConfigParser()
    config.read('settings.ini')

    # parameters recovering
    # features domain
    fdom = config.get('section_b','fdom')
    sampling_freq = config.getfloat('section_b','sampling_freq')
    # epoch half size as int
    epk_half_sizei = config.getint('section_a','epk_half_size')

    # frequencies banks
    frequency_bands = eval(config.get('section_a','frequency_bands'))

    # best setting recovering
    best_setting = config.get('section_c','best_setting').split(',')

    if (best_setting[0] == 'None'):
        print('please run training_pipeline script before testing!!')
    else:
        # freatures domain
        fdom = best_setting[0]
        # reduction  procedure
        redux_proc = best_setting[1]
        # classifiers
        clf_type = best_setting[2]


    # Raw data recovering
    print('Data loading..')
    r_data = load_raws_within_dir(data_directory_path)

    # BUILDING ARTIFICIAL EQUALLY SPACED WINDOWS OVER PSEUDO EVENTS
    windows = []
    for raw in r_data:
        windows.append(windower(raw,0,-epk_half_sizei/sampling_freq, epk_half_sizei/sampling_freq))


    # FEATURES COMPUTATION
    features_set = None
    if (fdom == 'time') or (fdom == 'time_frequency'):
        print('######################## Time Domain Features - computations -')
        tdf = extract_td_features_from_epks(windows)

        # data formatting/reshaping
        rtdf = reshape_numpy(tdf)

        # standardization
        rtdf_std  = []
        for data in rtdf:
            rtdf_std.append(standardize_data(data))
        features_set = rtdf_std


    if (fdom == 'frequency') or (fdom == 'time_frequency'):
        # frequency domain coefficients computing
        print('########################Frequency domain coefficients computation..')
        print(type(frequency_bands))
        fd_coeffs = band_filter(windows, frequency_bands)

        print('######################## Frequency Domain Features - computations -')
        fdf = []
        for dec in fd_coeffs:
            fdf.append(svm_features(dec))

        # data formatting (reshaping)
        rfdf = reshape_numpy(fdf)

        # standardization
        rfdf_std = []
        for data in rfdf:
            rfdf_std.append(standardize_data(data))

        features_set = rfdf_std


    if fdom == 'time_frequency':
        # time and frequency domain features concatenation
        rtfdf = []
        for tf,ff in zip(rtdf,rfdf):
            print(tf.shape,ff.shape)
            rtfdf.append(np.concatenate((tf,ff),axis=1))

        # standardization_events_to_raws
        rtfdf_std = []
        for features in rtfdf:
            rtfdf_std.append(standardize_data(features))

        features_set = rtfdf_std

    # DIMENSION REDUCTION
    redux_set = []
    for features in features_set:
        if redux_proc == 'pca':
            redux_set.append(pca(features, 2))
        elif redux_proc == 'ica':
            redux_set.append(ica(features, 2))
        #elif redux_proc == 'lda':
        #    redux = eest.lda(fset, 2, labset)
        else:  # no reduction -> ident
            redux_set.append(ident(features))


    # CLASSIFICATION

    # classifier selection
    n_classes = 2
    if clf_type == 'kmeans':
        clf = KMeans(n_clusters=n_classes)
    #elif clf_type == 'svm':
    #    # SVM- support vector machine
    #    clf = svm.SVC()
    elif clf_type == 'hc':  # hierarchical clustering
        clf = AgglomerativeClustering(n_clusters=n_classes, affinity='euclidean', linkage='ward')
    elif clf_type == 'if':  # isolation forest
        clf = IsolationForest()
    elif clf_type == 'em':
        # n_components shall be chosen via bic criterion
        # cv_type: full(default)/spherical/tied/dag
        clf = GaussianMixture(n_components=n_classes, covariance_type='full')
    elif clf_type == 'ap':  # affinity propagation
        clf = AffinityPropagation(random_state=5, max_iter=1000)  # convergence issues might need tuning
    elif clf_type == 'bgm':  # BayesianGaussianMixture
        clf = BayesianGaussianMixture(n_components=n_classes, max_iter=200)
    else:  # error handling (default behaviour) todo
        print('lkajdflkj----- bad clf_type')
        clf = None

    # PREDICTION
    predicted = []
    for features in redux_set:
        clf.fit(features[0])
        predicted.append(clf.predict(features[0]))

    # RAW OBJECT: EVENT ADDITION
    pkldump(r_data,'r_data.txt')
    pkldump(windows,'windows.txt')
    pkldump(predicted,'predidcted.txt')

    tagged = add_events_to_raws(predicted,windows,r_data)



    a = 11

if __name__ == '__main__':
    import sys
    from pathlib import Path
    from os.path import isfile, isdir

    datapath = None
    init_file = None
    datapath = 'C:/Users/oreligieux/gbcf_gee/raws_dir/test'
    init_file = 'C:/Users/oreligieux/gbcf_gee/SpikeTagger/settings.ini'

    """
    if len(sys.argv) > 3:
        raise RuntimeError("Too many arguments, max is 2.")

    if len(sys.argv) > 2:
        init_file = sys.argv[2]

    if len(sys.argv) > 1:
        if not init_file:
            init_file = (Path(input("Provide the config file's path:")))
        datapath = sys.argv[1]

    if len(sys.argv) == 1:
        datapath = str(Path(input("Provide the data's path:")))
        init_file = (Path(input("Provide the config file's path:")))

    if not isdir(datapath):
        raise NotADirectoryError("Data path {} does not exist".format(datapath))
_events_to_raws
    if not isfile(init_file):
        raise FileNotFoundError("{} does not exist.".format(init_file))
    """

    main(datapath, init_file)