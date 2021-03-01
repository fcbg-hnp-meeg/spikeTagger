# add_events_to_raws.py
import numpy as np

def add_events_to_raw(predicted,epochs,raw,pref_ch,win_half_size):
    print('laskjdf',predicted.shape,epochs.get_data().shape,'lkjsdf')
    print(sum(predicted==1),predicted[predicted==1].shape)
    #events = np.zeros((sum(predicted==1),3))
    events = epochs.events[predicted==1,:]
    wins = epochs.get_data()[predicted==1,:,:]
    for e,w in zip(events,wins):
        #print(e.shape,w.shape,e[0],max(w[pref_ch,:]),np.argmax(w[pref_ch,:]),np.argmax(abs(w[pref_ch,:])))
        e[0] = e[0]-win_half_size+np.argmax(abs(w[pref_ch,:]))

    raw.load_data().add_events(events,stim_channel='STI',replace=True)
    return(raw)

def add_events_to_raws(predictions,epochs,raws,pref_chs,win_half_size):
    tagged = []
    for p,e,r,ch in zip(predictions,epochs,raws,pref_chs):
        tagged.append(add_events_to_raw(p,e,r,ch,win_half_size))

    return(tagged)