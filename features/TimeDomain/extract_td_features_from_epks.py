# extract_features_from_epks.py
import numpy as np
import scipy.stats as stats

def data_mvsk(dat):  # inval \in np.nparray
    """

    Args:
        dat (np.array): An array of shape(epochs,channels,values); the data value from which
        features are to be computed.

    Returns (np.array): An array of shape(epochs,channels,features).
                        Where the features are the first four moments  (mean,variance,skewness,kurtosis)

    """
    print('----------------__data_mvsk__',dat.shape)
    out = []

    # (a)mean
    m_avg = np.mean(dat, -1)  # <- ------------------------------- TODO verif dim !!!

    # (b)variance
    m_var = np.var(dat, -1)

    # (d)skewness (3rd moment)
    m_skew = stats.skew(dat, -1)

    # (e)kurtosis (4th moment)
    m_kurt = stats.kurtosis(dat, -1)

    out.append(m_avg.T)
    out.append(m_var.T)
    out.append(m_skew.T)
    out.append(m_kurt.T)  # -3:Fisher's def ¦¦ 0 Pearson's def

    arr = np.asarray(out)

    return ((arr.T))

def extract_features_from_epk(mne_epk):
    print('------------------extract_features_from_epk')
    out = data_mvsk(mne_epk.get_data(picks='eeg'))
    return(out)

def extract_td_features_from_epks(mne_epk_set):
    out = []
    for e in mne_epk_set:
        out.append(extract_features_from_epk(e))

    return(out)
