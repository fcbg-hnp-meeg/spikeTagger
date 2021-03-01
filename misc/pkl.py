import pickle

def pkldump(obj,file):
    ofile = open(file,'wb')
    pickle.dump(obj,ofile)
    ofile.close()

def pklload(file):
    ifile = open(file, 'rb')
    out = pickle.load(ifile)
    ifile.close()

    return(out)