# plot_mosaic.py
import numpy as np
import  seaborn as sns
import matplotlib.pyplot as plt

def recover_data_4mosaicplot(collection,index,nb_rows,nb_cols):
    """
    This function works in pair with 'heatmap_plot' function.
    It formats data (collection) into a matrix of nb_rows (#reduction procedures)
    by nb_cols (# ML algorithms) containing values indexed @ the 'index' position of the
    collection entries

    Args:
        collection:
        index (int): position of the relevant value
        nb_rows (int): number of raws ==
        nb_cols:

    Returns:

    """
    print('----------------- recover_data_4mosaicplot --------------------')
    # nb configuration
    nb_conf = len(collection[0])
    # nb subjects
    nb_subject = len(collection)
    means = np.zeros(nb_conf)
    for c in range(nb_conf): # loop on configurations
        for m in range(nb_subject): # loop on subjects
            info = collection[m][c][0]
            print('m,c',m,c,info,info[index])
            means[c] = means[c]+info[index]

    print(means.shape)
    means_rsh =np.reshape(means,(nb_rows,nb_cols))
    print(means_rsh.shape)

    return(means_rsh/nb_subject)

def heatmap_plot(data,metric_title,xlabs,ylabs):
    """
    This function plots a heatmap  of data:
    Warning: order is top-down left-to-right

    Args:
        data (matrix): data
        metric_title (string): plot title
        xlabs list<string>: x labels
        ylabs list<string>: y labels

    Returns:

    """
    print('----------------- mosaic_plot --------------------')
    # plotting
    #fig,(ax,ax2) = plt.subplots(ncols=2)
    #fig.subplots_adjust(wspace=.01)
    sns.heatmap(data,xticklabels=xlabs,yticklabels=ylabs,annot=True,cbar=False,cmap=sns.cm.rocket_r) # cmap=Blue?
    #ax.title.set_text('time')
    #sns.heatmap(data[1],xticklabels=xlabs,yticklabels=False,annot=True,ax=ax2,cmap='Blue')
    #ax2.tick_params(left=False)
    #ax2.set_ylabel('')
    #ax2.title.set_text('time-freq')

    ##fig.suptitle(metric_title)
    plt.savefig(metric_title+'.png')
    plt.show()