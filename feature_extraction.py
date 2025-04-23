import pandas as pd
import soundfile as sf
import numpy as np
import os
import librosa
from features_computation import *

# Load your original DataFrame
df = pd.read_csv('D:\\CMP_6\\Neural_Networks\\Project\\filtered_data_labeled.tsv', sep='\t')

if __name__ == "__main__":
    # Paths to your audio files
    folder = "D:\\CMP_6\\Neural_Networks\\audio_batch_20"
    file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]

    i = 0
    batch_num = 1
    audio_features_list = []
    batch_size = 10000  # Save every 10 files, you can change it

    for path in file_paths:
        print(f"Processing: {path}")
        try:
            signal, sample_rate = librosa.load(path, sr=None)
            features = extract_all_features(signal, sample_rate)
            features["path"] = os.path.basename(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            features = None

        if features is not None:
            audio_features_list.append(features)
            i += 1

        print(f"Processed {i} files.")

        # Save every batch_size files
        if i % batch_size == 0 and i != 0:
            audio_df = pd.DataFrame(audio_features_list)
            merged_df = pd.merge(df, audio_df, on='path', how='left')
            
            save_path = f'merged_audio_features_batch_{batch_num}.csv'
            merged_df.to_csv(save_path, index=False)
            print(f"Saved batch {batch_num} to {save_path}")
            
            audio_features_list = []  # Clear for next batch
            batch_num += 1

    # Save any remaining files after loop ends
    if audio_features_list:
        audio_df = pd.DataFrame(audio_features_list)
        merged_df = pd.merge(df, audio_df, on='path', how='left')
        
        save_path = f'merged_audio_features_batch_{batch_num}.csv'
        merged_df.to_csv(save_path, index=False)
        print(f"Saved final batch {batch_num} to {save_path}")
