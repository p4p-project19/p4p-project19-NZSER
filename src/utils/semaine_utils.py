import os
import glob
import pandas as pd
from librosa import load

signals = []
SAMPLING_RATE = 16000
''' 
One time use only.
'''
# Rename folders in semaine that dont have at least 2 text files inside with a "NA"
def prune_semaine():
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    semaine_path = root + "/data/semaine/sessions"
    for folder in os.listdir(semaine_path):
        if len(glob.glob(semaine_path + "/" + folder + "/*.txt")) < 2:
            os.rename(semaine_path + "/" + folder, semaine_path + "/" + folder + "_NA")
    return


def load_semaine():
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    semaine_path = root + "/data/semaine/sessions"
    for folder in os.listdir(semaine_path):
        if folder.endswith("_NA"):
            continue
        else:
            
            for file in glob.glob(semaine_path + "/" + folder + "/*.wav"):
                current_signal = load(file, sr=SAMPLING_RATE)
                current_signal = [current_signal[0]]
                signals.append(current_signal)
    return