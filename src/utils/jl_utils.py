import os
import glob
import pandas as pd
from librosa import load
from utils.audio_utils import get_audio_chunks

# Constants
SAMPLING_RATE = 16000
IS_JL = True

def load_jl():
    """
    Load the JL-Corpus dataset and annotations.
    """

    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # load all wav files in a directory into a list
    files = glob.glob(root + "/data/jl/*.wav")
    signals = [[] for i in range(len(files))]

    for i, file in enumerate(files):
        # Load the signal resampled at 16kHz
        current_signal = load(file, sr=SAMPLING_RATE)
        current_signal = [current_signal[0]]
        signals[i].append(current_signal)

    # Load csv files with annotations
    csv_files = glob.glob(root + "/data/jl/*.csv")
    
    # f1 = female1, m2 = male2
    f1_aro_df = pd.read_csv(csv_files[0])
    f1_val_df = pd.read_csv(csv_files[1])
    m2_aro_df = pd.read_csv(csv_files[2])
    m2_val_df = pd.read_csv(csv_files[3])
    
    f1_aro_df = f1_aro_df[['bundle', 'labels']]
    f1_aro_df = f1_aro_df.rename(columns = {'labels': 'arousal'})
    f1_val_df = f1_val_df[['bundle', 'labels']]
    f1_val_df = f1_val_df.rename(columns = {'labels': 'valence'})

    m2_aro_df = m2_aro_df[['bundle', 'labels']]
    m2_aro_df.rename(columns = {'labels': 'arousal'})
    m2_val_df = m2_val_df[['bundle', 'labels']]
    m2_val_df.rename(columns = {'labels': 'valence'})

    f1_mer_df = pd.concat([f1_aro_df, f1_val_df], axis=1)
    m2_mer_df = pd.concat([m2_aro_df, m2_val_df], axis=1)
    pass
        
    