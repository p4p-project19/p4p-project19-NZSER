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
        aro_df = aro_df.drop('FF3', axis=1)
        aro_df['bundle'] = os.path.basename(csv_file).split('.')[0]
        aro_dfs.append(aro_df)
        

    val_dfs = []
    for csv_file in val_csv_files:
        val_df = pd.read_csv(csv_file, delimiter=';')
        val_df = val_df.drop('FF3', axis=1)
        val_df['bundle'] = os.path.basename(csv_file).split('.')[0]
        val_dfs.append(val_df)