# training_pipeline.py
import sys
import numpy as np
from configparser import ConfigParser
from pathlib import Path

from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering  # No predict fun.. only fit_predict
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn import svm
from misc import load_raws_within_dir
from  misc import get_labels_from_raws


from processing import add_spikeless_events_to_raws
from processing import get_epochs_from_raws

from preprocessing import windower

from features.TimeDomain import  extract_td_features_from_epks
from features.FrequencyDomain import svm_features
from util import reshape_numpy
from util import order_by_perf
from preprocessing import standardize_data
from processing import band_filter
from redux import ident,pca,ica,lda

from classify import train_clf
from plot import recover_data_4mosaicplot,heatmap_plot

def main(data_directory_path,init_file):
    print('DATA_DIRECTORY :',data_directory_path)
    print('CONFIGURATION_FILE:',init_file)

    # settings recovering
    # via settings.ini file
    print('Parameters recovering..')
    config = ConfigParser()
    config.read('settings.ini')

    # parameters recovering
    # features domain
    fdom = config.get('section_b','fdom')

    # reduction procedures to be applied
    redux_procedures = config.get('section_b','redux_procedures').split(',')

    # classifiers to be used
    classifiers = config.get('section_b','classifiers').split(',')


    # epoch half size as int
    epk_half_sizei = config.getint('section_a','epk_half_size')
    # epoch half size as float
    epk_half_sizef = epk_half_sizei/1000

    # number of folds
    k_folds = config.getint('section_b','k_folds')

    # frequencies banks
    frequency_bands = eval(config.get('section_a','frequency_bands'))


    # Raw data recovering
    print('Data loading..')
    #r_data = msc.load_raws_from_dir(data_directory_path)
    r_data = load_raws_within_dir(data_directory_path)
    print('Spikeless events tagging..')
    r_01 = add_spikeless_events_to_raws(r_data,epk_half_sizei)

    # labels revovering from raw data
    labels = get_labels_from_raws(r_01)

    #'''
    # Epochs computation
    print('Epoch building..')
    epks = get_epochs_from_raws(r_01,epk_half_sizef)


    # FEATURES COMPUTATION
    if (fdom == 'time') or (fdom == 'time_frequency'):
        print('######################## Time Domain Features - computations -')
        tdf = extract_td_features_from_epks(epks)
        #msc.pkldump(tdf,'td_features.pkl')
        #tdf = msc.pklload('td_features.pkl')

        # data formatting (reshaping)
        rtdf = reshape_numpy(tdf)

        # standardization
        rtdf_std  = []
        for data in rtdf:
            rtdf_std.append(standardize_data(data))

    if (fdom == 'frequency') or (fdom == 'time_frequency'):
        # frequency domain coefficients computing
        print('########################Frequency domain coefficients computation..')
        print(type(frequency_bands))
        fd_coeffs = band_filter(epks, frequency_bands)
        #msc.pkldump(fd_coeffs, 'fd_coeffs.pkl')
        #fd_coeff = msc.pklload('fd_coeffs.pkl')

        print('######################## Frequency Domain Features - computations -')
        fdf = []
        for dec in fd_coeffs:
            fdf.append(svm_features(dec))

        #msc.pkldump(fdf,'fd_features.pkl')
        #fdf = msc.pklload('fd_features.pkl')

        # data formatting (reshaping)
        rfdf = reshape_numpy(fdf)

        # standardization
        rfdf_std = []
        for data in rfdf:
            rfdf_std.append(standardize_data(data))

    if fdom == 'time_frequency':
        # time and frequency domain features concatenation
        rtfdf = []
        for tf,ff in zip(rtdf,rfdf):
            print(tf.shape,ff.shape)
            rtfdf.append(np.concatenate((tf,ff),axis=1))

        # standardization
        rtfdf_std = []
        for features in rtfdf:
            rtfdf_std.append(standardize_data(features))

    # DIMENSION REDUCTION
    redux = []
    for fset,labset in zip(rtfdf_std,labels): # loop on subjects features sets
        sbj_redux = []
        for rdx in redux_procedures: # loop on reduction procedure to apply
            if rdx == 'pca':
                sbj_redux.append(pca(fset,2))
            elif rdx == 'ica':
                sbj_redux.append(ica(fset,2))
            elif rdx == 'lda':
                sbj_redux.append(lda(fset, 2, labset))
            else:# no reduction -> ident
                sbj_redux.append(ident(fset, 2, labset))
        redux.append(sbj_redux)

    #msc.pkldump(redux,'features_reductions.pkl')
    #redux = msc.pklload('features_reductions.pkl')

    # CLASSIFIER TRAINING
    res = []
    for subject,labels_set in zip(redux,labels): # loop on sujbects & according labels
        print('#####')

        tmp = []
        for clf in classifiers:
            print('~~~~~~~')
            for rdx in subject:
                print('-----') #todo warning on y_pred¦y_true labels
                clf_list = train_clf(features_set = rdx[0],
                               clf_type = clf,
                               n_classes=2,
                               labels_set = labels_set,
                               k_folds=k_folds,
                               additional_params=[rdx[2]])
                tmp.append(clf_list)
        res.append(tmp)

    #msc.pkldump(res,'train_pipe_res.pkl')
    #'''
    #res = msc.pklload('train_pipe_res.pkl')

    # BEST SETTINGS RECOVERING % RECORDING
    ordered = order_by_perf(res,1)

    # MOSIAC PLOT
    hmd_data = recover_data_4mosaicplot(res,1,len(classifiers),len(redux_procedures))
    heatmap_plot(hmd_data,'lkjdf',xlabs=redux_procedures,ylabs=classifiers)
    ad = 19


if __name__ == '__main__':

    from pathlib import Path
    from os.path import isfile, isdir

    datapath = None
    init_file = None

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

    if not isfile(init_file):
        raise FileNotFoundError("{} does not exist.".format(init_file))

    main(datapath, init_file)