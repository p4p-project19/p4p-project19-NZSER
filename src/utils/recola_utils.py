import os
import glob
import pandas as pd
from librosa import load

# Constants
SAMPLING_RATE = 16000

def load_recola():
    """
    Load the Recola dataset and annotations
    """
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # Load csv files with annotations
    aro_csv_files = glob.glob(root + "/data/recola/RECOLA-Annotation/emotional_behaviour/arousal/*.csv")
    val_csv_files = glob.glob(root + "/data/recola/RECOLA-Annotation/emotional_behaviour/valence/*.csv")

    aro_dfs = []
    for csv_file in aro_csv_files:
        aro_df = pd.read_csv(csv_file, delimiter=';')

        aro_df.insert(0, 'bundle', os.path.basename(csv_file).split('.')[0]) # Add filename to dataframe
        aro_df = aro_df.drop('FF3', axis=1) # Drop FF3 annotator due to lack of range
        # Rename columns for arousal
        aro_df = aro_df.rename(columns = {'FM1 ': 'A1', 'FM2 ':'A2', 'FM3 ':'A3', 'FF1 ': 'A4', 'FF2 ': 'A5'})
        
        aro_dfs.append(aro_df)
        
    # Load valence csv files and concatentate arousal and valence dataframes
    mer_dfs = []
    for i, csv_file in enumerate(val_csv_files):
        val_df = pd.read_csv(csv_file, delimiter=';')

        val_df = val_df.drop('FF3', axis=1)
        # Rename columns for valence
        val_df = val_df.rename(columns = {'FM1 ': 'V1', 'FM2 ':'V2', 'FM3 ':'V3', 'FF1 ': 'V4', 'FF2 ': 'V5'})
        
        mer_dfs.append(pd.concat([aro_dfs[i], val_df], axis=1))
