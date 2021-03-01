# get_labels_from_raws.py
import mne

def get_labels_from_raw(raw):
    """
     Returns the events labels of the raw obj.
    Args:
        raw (mne.Raw): A mne.Raw obj. from wich one wants to get the  events labels

    Returns (np.array<int>):
        A np.array of (int) labels

    """
    print('------------------ get labels from raw ------------', type(raw))
    out = mne.find_events(raw)
    # print('......:',out.shape)
    return (out[:, 2])


def get_labels_from_raws(raws_list):
    """
    This function mainly calls 'get_labels_from_raw' to get labels associated to mne.Raw obj. in the list
    given as parameter.
    Args:
        raws_list (list<mne.Raw):  A list of mne.Raw obj. from which the labels are to be ..

    Returns (list<np.array<int>>:
        A list of labels corresponding to the mne.Raw obj.

    """
    print('------------------ get labels from rawS ------------')
    labels = list(map(get_labels_from_raw, raws_list))

    return (labels)
