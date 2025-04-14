import pandas as pd
import soundfile as sf
import numpy as np
import os
import librosa
from features_computation import *

# Load your original DataFrame
df = pd.read_csv('D:\\CMP_6\\Neural_Networks\\Project\\merged_audio_features_filtered.csv')
if __name__ == "__main__":
    folder = "D:\\CMP_6\\Neural_Networks\\audio_batch_20"
    file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]

    i = 0
    audio_features_list = []
    merged_df = None   
    batch_size = 10000

    for path in file_paths:
        print(path)
        try:
            signal, sample_rate = librosa.load(path, sr=None)
            features = extract_all_features(signal, sample_rate)
            features["path"] = os.path.basename(path)
        except Exception as e:
            print(e)
            features = None

        if features is not None:
            audio_features_list.append(features)
            i += 1
            # os.remove(path)

        print(i)

        if i % batch_size == 0 and i != 0:
            audio_df = pd.DataFrame(audio_features_list)
            if merged_df is None:
                merged_df = pd.merge(df, audio_df, on='path', how='left')
            else:
                merged_df = pd.concat([merged_df, audio_df], ignore_index=True)

            merged_df.to_csv('merged_audio_features_2.csv', index=False)
            audio_features_list = []