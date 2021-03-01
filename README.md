# spikeTagger
Tag spikes on unlabeled data 

This module contains 2 scripts.


A:
Find_best_setting.py:

This 1rst script compute performances of different setting that are specified in the setting.ini file.
Given these attributes, it will compute the performances of the algorithm to allow the best setting election, 
and recorded them in a list (res := results)

Additionally, it will plot the perfomances results in a heatmap plot.

usage: 
python find_best_setting.py 
(script will ask for data_directory and setting_file paths)


Input => 
i) directory path to the raw_fif format file on wich the algorithm are to be trained
ii) The path to the setting.ini file that is to be use

output => 

B:
spike_tagger.py

This 2nd script use best setting to tag the 'eeg' signal.
Input => 
i) directory path to the raw_fif format file on wich the algorithm are to be trained
ii) The path to the setting.ini file that is to be use

usage:
python spike_tagger.py
(default setting is specified in the best_setting field of the setting.ini file)
