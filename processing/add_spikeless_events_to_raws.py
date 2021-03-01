import mne
import numpy as np
from functools import partial

def add_spikeless_events_to_events_array(events_array, epoch_half_size, overlap=0, margin=10):
    """
    This function adds events (spikeless) to the MNE events array.
    It fills space between events with equaly space new event and updates
    the events array accordingly
    Args:
        events_array (np.array(n,3)): Original event array
        epoch_half_size(int): holf size of epoch in  [ms]
        overlap (int): [ms]
        margin (int): [ms]

    Returns:

    """
    print('--------------------add_spike-less_events_to_events_array--------------------')
    epoch_size = 2 * epoch_half_size + 1 + margin
    spikeless_events = np.empty((1, 3))

    # space BEFORE first event treatment # todo

    for e in range(events_array.shape[0] - 1):
        current = events_array[e, 0] + epoch_half_size
        while current + epoch_size < events_array[e + 1, 0]:
            new_events = np.array([current + epoch_half_size + 1, 0, 2])
            spikeless_events = np.vstack((spikeless_events, new_events))
            current += epoch_size

            # todo above lines needed?
        if events_array[e + 1, 0] - spikeless_events[-1, 0] < epoch_size:  # todo maybe add test on spikless_event.size
            # delete last element
            spikeless_events = spikeless_events[0:-1, :]

    # space AFTER last event treatment # todo

    events_array = np.vstack((events_array, spikeless_events[1:, :]))
    events_array = events_array[np.argsort(events_array[:, 0]), :]

    return (events_array)

def add_spikeless_events_to_raw(raw_obj, epoch_half_size, overlap=0, margin=10):  # todo overlap treatment
    """
    Given a mne.raw obj., this function add events (spikeless) where there is enough  space.
    Args:
        raw_obj (mne.Raw): the Raw obj. on which to add events
        epoch_half_size (int): Size of window/epoch
        overlap:
        margin:

    Returns (mne.Raw): A raw obj. filled with new spikeless events

    """
    print('--------------------add_spike-less_events_to_raw--------------------')
    r = raw_obj.load_data().copy()
    #r.load_data()
    base_events = mne.find_events(r)

    base_events = add_spikeless_events_to_events_array(base_events, epoch_half_size, overlap, margin)
    r.load_data()
    r.add_events(base_events, stim_channel='STI', replace=True)


    return (r)


def add_spikeless_events_to_raws(raw_obj_list, epoch_half_size, overlap=0, margin=10):
    """
    This function add spikeless events to a mne.Raw obj. list
    Args:
        raw_obj_list (listy<mne.Raw>): a list of mne.Raw obj. on which spikeless events are
                                      to be added
        epoch_half_size (int): Epoch size/2
        overlap:
        margin:

    Returns: A list of mne.Raw obj. filled with spikeless events

    """
    print('--------------------add_spike-less_events_to_rawS--------------------')
    out = list(map(partial(add_spikeless_events_to_raw,
                           epoch_half_size=epoch_half_size,
                           overlap=overlap,
                           margin=margin),
                   raw_obj_list))
    return (out)
