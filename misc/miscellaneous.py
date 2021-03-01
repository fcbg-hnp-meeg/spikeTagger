# miscellaneous.py
import mne
import pickle
from pathlib import Path

def load_raws_from_dir(path_to_dir):
    print('load_raws_from_dir:',path_to_dir)
    raws = []
    path = Path(path_to_dir)

    files_names = [e for e in path.iterdir() if e.is_file()]
    for fn in files_names:
        print(fn)
        raws.append(mne.io.read_raw_fif(fn))

    return(raws)
def pkldump(obj,file):
    ofile = open(file,'wb')
    pickle.dump(obj,ofile)
    ofile.close()

def pklload(file):
    ifile = open(file, 'rb')
    out = pickle.load(ifile)
    ifile.close()

    return(out)
