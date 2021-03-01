#svm_features4all.py
import numpy as np
def means_abs_coefs(coefs):
    #print('\t-----------------means_abs_coefs-----------------')
    return(np.mean(abs(coefs),-1))

# [II]:average power of the wavelet coeff (for each sub-band)
# pow_i = ¦w_i¦^2; w_i coef_i
def avg_coef_pow(coefs):
    #print('\t-----------------avg_coefs_pow-----------------')
    tmp = abs(coefs)
    tmp = np.multiply(tmp,tmp)
    return(np.mean(tmp,-1))

# [III]: stand. dev. of coef (for each sub-band)
def coef_std(coefs):
    #print('\t-----------------coefs_std-----------------')
    return(np.var(coefs,-1))

# [IV]: adj. sub-band abs.mean ratio
def adj_absmean_ratio(left,right): ## todo bounds management
    print('\t-----------------adj_means_ratio-----------------')
    return(left/right)

def svm_paper_features3(psd_coefs):
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

def svm_features(bank_coefs_list): #todo checkout map(..)
    """

    :param bank_coefs_list (list<list>): list of
    :return:
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