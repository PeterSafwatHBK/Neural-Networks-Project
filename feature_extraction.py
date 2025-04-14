import pandas as pd
import soundfile as sf
import numpy as np
import os
import librosa
from features_computation import *

# Load your original DataFrame
df = pd.read_csv('D:\\CMP_6\\Neural_Networks\\Project\\merged_audio_features_2.csv')
l = df["energy"].dropna().tolist()
print(l)
if __name__ == "__main__":
    # Paths to your audio files
    folder = "D:\\CMP_6\\Neural_Networks\\audio_batch_20"
    file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]

    # Extract features
    i = 0
    audio_features_list = []
    merged_df = None   # Initialize here
    batch_size = 10000

    for path in file_paths:
        print(path)

        if os.path.basename(path) in l:
            os.remove(path)

        # try:
        #     signal, sample_rate = librosa.load(path, sr=None)
        #     features = extract_all_features(signal, sample_rate)
        #     features["path"] = os.path.basename(path)
        # except Exception as e:
        #     print(e)
        #     features = None

        # if features is not None:
        #     audio_features_list.append(features)
        #     i += 1
        #     # os.remove(path)

        # print(i)

        # # Save every batch_size files
        # if i % batch_size == 0 and i != 0:
        #     audio_df = pd.DataFrame(audio_features_list)
        #     if merged_df is None:
        #         # First time: merge
        #         merged_df = pd.merge(df, audio_df, on='path', how='left')
        #     else:
        #         # After first merge: just update
        #         merged_df = pd.concat([merged_df, audio_df], ignore_index=True)

        #     merged_df.to_csv('merged_audio_features_2.csv', index=False)
        #     audio_features_list = []  # Clear the list for next batch


    # # After the loop, check if any remaining features are unsaved
    # if audio_features_list:
    #     audio_df = pd.DataFrame(audio_features_list)
    #     if merged_df is None:
    #         merged_df = pd.merge(df, audio_df, on='path', how='left')
    #     else:
    #         merged_df = pd.concat([merged_df, audio_df], ignore_index=True)
    #     merged_df.to_csv('merged_audio_features.csv', index=False)
