# order_by_perf.py
import numpy as np
def get_index_info_list(info_list,index_info):
    """

    Args:
        info_list:
        index_info:

    Returns:

    """
    out = []
    for i in info_list:
        #print(index_info,len(info_list),len(i),i,i[0][index_info])
        out.append(i[0][index_info])
    return(out)
def perf_sort(info_list,perf_index):
    sorting_dat = get_index_info_list(info_list,perf_index)
    order = np.argsort(np.asarray(sorting_dat))
    print(order)

    # X(((!!!
    out = []
    for i in order:
        out.append(info_list[i])
    #out = sorted(info_list,key=lambda order:order)
    return(out)
def order_by_perf(infos_list,perf_index):
    print('------------- order_by_perf-------------')
    out = []
    for  m in infos_list: #loop on mice
        out.append(perf_sort(m,perf_index))

    return(out)
