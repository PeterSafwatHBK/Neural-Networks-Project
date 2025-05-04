import pandas as pd
import soundfile as sf
import numpy as np
import os
import librosa
from features_computation import *
from signal_denoising import denoise_signal
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Load your original DataFrame
df = pd.read_csv(
    'filtered_data_labeled.tsv', sep='\t')

if __name__ == "__main__":
    # Paths to your audio files
    folder = "filtered_clips"
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument('--num_files', type=int, default=None, 
                        help="Number of files to process. If not specified, process all files.")
    args = parser.parse_args()

    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]
    file_paths = all_files[:args.num_files] if args.num_files else all_files

    # i = 0
    audio_features_list = []
    # batch_size = 10000  # Save every 10 files, you can change it
    def process_file(path):
        try:
            signal, sample_rate = librosa.load(path, sr=None)
            signal = denoise_signal(signal, sample_rate)
            features = extract_all_features(signal, sample_rate)
            features["path"] = os.path.basename(path)
            return features
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    results = []
    with ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(process_file, file_paths), total=len(file_paths), desc="Processing files"):
            results.append(result)

    # Filter out None results
    audio_features_list = [res for res in results if res is not None]

    audio_df = pd.DataFrame(audio_features_list)
    
    # print(audio_df)
    merged_df = pd.merge(df, audio_df, on='path', how='left')

    save_path = "new_features.csv"
    merged_df.to_csv(save_path, index=False)
    print(f"Saved features to {save_path}")


    # # Save any remaining files after loop ends
    # if audio_features_list:
    #     audio_df = pd.DataFrame(audio_features_list)
    #     merged_df = pd.merge(df, audio_df, on='path', how='left')

    #     save_path = f'merged_audio_features_batch_{batch_num}.csv'
    #     merged_df.to_csv(save_path, index=False)
    #     print(f"Saved final batch {batch_num} to {save_path}")
