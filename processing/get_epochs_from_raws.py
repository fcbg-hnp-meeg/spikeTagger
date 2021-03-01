# get_epochs_from_raws.py
import mne

def get_epochs_from_raw(raw_obj, epk_half_size, labels=False):
    """
    A function to get Epoch out of a Raw obj
    Args:
        raw_obj:
        epk_half_size:
        labels:

    Returns: Related epochs

    """
    print('--------------------epochs_from_raw--------------------')
    epk = mne.Epochs(raw_obj, mne.find_events(raw_obj),
                     tmin=-epk_half_size, tmax=epk_half_size,
                     preload=True)

    return (epk)


def get_epochs_from_raws(raw_list, epk_half_size):
    """
    Given a list of mne.Raw, this function gets a list of related Epochs
    Args:
        raw_list:
        epk_half_size:

    Returns: A list of mne.Epoch

    """
    epk_list = []
    for r in raw_list:
        epk_list.append(get_epochs_from_raw(r.load_data(), epk_half_size))
    return (epk_list)

