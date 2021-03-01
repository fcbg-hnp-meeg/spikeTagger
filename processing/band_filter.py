# band_filter.py
from mne.time_frequency import psd_multitaper

def turnaround_mne_psd(epk, freq_banks):
    psds = []
    print('---------', type(epk))
    for fb in freq_banks:
        decomposition, freqs = psd_multitaper(epk, fmin=fb[0], fmax=fb[1])
        psds.append(decomposition)

    return (psds)


def band_filter(epks_list, freq_banks):
    """
    This function returns a list of coefficient according to psd decomposition
    Args:
        epks_list:
        freq_banks:

    Returns:

    """
    filtered = []
    print('bnk-filtr-epk', type(epks_list))
    # loop on mice epk's
    for e in epks_list:
        print('applying psd to epock', type(e))
        filtered.append(turnaround_mne_psd(e, freq_banks))

    return (filtered)
