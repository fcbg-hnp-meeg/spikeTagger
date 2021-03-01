# extract_features_from_psd_coefficients.py
import numpy as np

def means_abs_coefs(coefs):
    """

    Args:
        coefs (list<float>): the coefficients list of the multi-taper in a single sub-band

    Returns: the mean of absolute value of the coeff.

    """
    #print('\t-----------------means_abs_coefs-----------------')
    return(np.mean(abs(coefs),-1))

# [II]:average power of the wavelet coeff (for each sub-band)
# pow_i = ¦w_i¦^2; w_i coef_i
def avg_coef_pow(coefs):
    """

    Args:
        coefs: ~

    Returns: The average coefficients power

    """
    #print('\t-----------------avg_coefs_pow-----------------')
    tmp = abs(coefs)
    tmp = np.multiply(tmp,tmp)
    return(np.mean(tmp,-1))

# [III]: stand. dev. of coef (for each sub-band)
def coef_std(coefs):
    """

    Args:
        coefs: ~

    Returns: the coefficients standard deviation

    """
    #print('\t-----------------coefs_std-----------------')
    return(np.var(coefs,-1))

def svm_paper_features3(psd_coefs):
    """

    Args:
        psd_coefs (list<float>): The coefficients of the multi-taper for a given sub-band

    Returns: array of values

    """
    #print('-----------------svm_paper_features-----------------',type(psd_coefs),psd_coefs.shape)
    mac = means_abs_coefs(psd_coefs)
    #print(mac.shape)
    if mac.any()==0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!mac.zero')
    #print(mac.shape)
    acp = avg_coef_pow(psd_coefs)
    #print(acp.shape)
    if acp.any()==0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!acp.zero')
    #print(acp.shape)
    cs  = coef_std(psd_coefs)
    if psd_coefs.shape[-1]!=1:
        print(cs.shape,'-')
    if cs.any()==0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!cs.zero')
    #print(cs.shape)

    return(np.dstack((mac,acp,cs)))

def extract_features_from_psd_coefficients(bank_coefs_list): #todo checkout map(..)
    """

    :param bank_coefs_list (list<list<float>>): list of coefficient for each sub-band coefficient for each sub-band
    :return: correcponding features values as a np.array
    """
    print('------------------svm_features_4all-----------------')

    out = []

    # loop on freq_banks(coefs)
    #res = np.empty((1846,16,3),dtype=np.float64)
    res = None

    # PASS.ONE: three first features
    i = 0
    for c in bank_coefs_list: # todo !!! prob with ..features3.coef_std (cs) when single value
        if (c.shape[-1] != 1):
            #print(i,'D',type(c),c.shape)
            i = i+1
            tmp = svm_paper_features3(c)
            #print('tmp.shape: ',tmp.shape,'-')

            out.append(tmp)
            if res is None:
                res = tmp
            else:
                res = np.concatenate([res,tmp],-1);
        else:
            print('-------Single value coefficients issue')

    # PASS.TWO: fourth feature:
    # ratio of adj_band.abs.means-ratio # todo

    #for d in range(1,len(bank_coefs_list)-1): #todo look for loop removal
    #    # adj.abs.mean.ratio
    #    aamr = res[:,:,d+1]/res[:,:,d-1] #<- b/a # warning a/b gives sporadic inf
    #    res = np.dstack((res,aamr))


    return(res)