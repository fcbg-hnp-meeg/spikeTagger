# windower.py
import mne
def windower(raw_obj, overlap, tmin, tmax):
    """
    This function is used to build sliding windows over the input signal

    Args:
        raw_obj (mne.io.Raw): raw recordings to be tagged
        overlap (float): percentage of window autorized overlapping
        half_winsize (int): wished window size

    Returns: mne.io.Epochs obj.

    """
    print('----------------- windower --------------------', type(raw_obj))
    r = raw_obj.copy().load_data()

    # set equally spaced events upon r
    acme_events = mne.make_fixed_length_events(raw_obj, id=5, start=.5, duration=(tmax-tmin)-overlap, first_samp=True, overlap=overlap)
    r.add_events(acme_events, stim_channel='STI', replace=True)

    # build epochs according to these r.events
    r.load_data().add_events(acme_events,stim_channel='STI',replace=True)
    epk = mne.Epochs(r, acme_events, tmin=tmin, tmax=tmax)

    return (epk)