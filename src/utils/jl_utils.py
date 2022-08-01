import os
import glob
import pandas as pd
from librosa import load

# Constants
SAMPLING_RATE = 16000
IS_JL = True

# f1 = female1, m2 = male2, df = dataframe
f1 = [] # Contains wav files and annotations for female1
m2 = [] # Contains wav files and annotations for male2
f1_dfs = []
m2_dfs = []

def load_jl():
    """
    Load the JL-Corpus dataset and annotations.
    """
    f1_dfs = []
    m2_dfs = []
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # # load all wav files in a directory into a list
    # files = glob.glob(root + "/data/jl/*.wav")
    # signals = [[] for i in range(len(files))]

    # for i, file in enumerate(files):
    #     # Load the signal resampled at 16kHz
    #     current_signal = load(file, sr=SAMPLING_RATE)
    #     current_signal = [current_signal[0]]
    #     signals[i].append(current_signal)

    # Load csv files with annotations
    csv_files = glob.glob(root + "/data/jl/*.csv")
    
    # f1 = female1, m2 = male2, df = dataframe
    f1_aro_df = pd.read_csv(csv_files[0])
    f1_val_df = pd.read_csv(csv_files[1])
    m2_aro_df = pd.read_csv(csv_files[2])
    m2_val_df = pd.read_csv(csv_files[3])
    
    f1_aro_df = f1_aro_df[['bundle', 'labels']]
    f1_aro_df = f1_aro_df.rename(columns = {'labels': 'arousal'})
    f1_val_df = f1_val_df['labels']
    f1_val_df = f1_val_df.rename('valence')

    m2_aro_df = m2_aro_df[['bundle', 'labels']]
    m2_aro_df = m2_aro_df.rename(columns = {'labels': 'arousal'})
    m2_val_df = m2_val_df['labels']
    m2_val_df = m2_val_df.rename('valence')

    f1_mer_df = pd.concat([f1_aro_df, f1_val_df], axis=1)
    m2_mer_df = pd.concat([m2_aro_df, m2_val_df], axis=1)
    # Remove duplicate bundle names and store in list
    f1_bdls = list(dict.fromkeys(f1_mer_df['bundle'].tolist()))
    m2_bdls = list(dict.fromkeys(m2_mer_df['bundle'].tolist()))

    for bdl in f1_bdls:
        bdl_df = f1_mer_df[f1_mer_df['bundle'] == bdl]
        f1_dfs.append(bdl_df)
        
    for bdl in m2_bdls:
        bdl_df = m2_mer_df[m2_mer_df['bundle'] == bdl]
        m2_dfs.append(bdl_df)
    print('Finished loading CSV files.')

    for df in f1_dfs:
        sig = load(root + "/data/jl/female1_all_a_1/" + df.iloc[0]['bundle'] + '_bndl/' + df.iloc[0]['bundle'] + ".wav", sr=SAMPLING_RATE)
        f1.append([df, sig])

    for df in m2_dfs:
        sig = load(root + "/data/jl/male2_all_a_1/" + df.iloc[0]['bundle'] + '_bndl/' + df.iloc[0]['bundle'] + ".wav", sr=SAMPLING_RATE)
        m2.append([df, sig])
    print('Finished loading WAV files.')
    