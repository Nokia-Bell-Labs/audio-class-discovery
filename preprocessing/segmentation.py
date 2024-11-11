'''© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear'''
import matplotlib.pyplot as plt
import librosa
import numpy as np
import ruptures as rpt  # our package

import pandas as pd

import os
import argparse
import warnings
from tqdm import tqdm

import soundfile as sf

def read_labels_df(class_display_name):
    filepath = os.path.join('/mnt/audioset', 'meta', f'{class_display_name}')

    if not os.path.exists(filepath):
        print(f"Directory for '{class_display_name}.csv' does not exist.")
        return None

    with open(filepath, 'r') as csv_file:
        #column_names = ['filename', 'start_seconds', 'end_seconds','labels', ('final_label')]  # YTID, start_seconds, end_seconds, positvie_labels
        labels_df = pd.read_csv(csv_file)
        print(f"{labels_df.head()}")

    return labels_df

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def segment_by_change_point(audio_dir, audio_file_name, output_audio_dir, output_image_dir):
    signal, sampling_rate = librosa.load(audio_dir + '/' + audio_file_name)
    if len(signal) == 0:
        print(f"file {audio_file_name} is empty.")
        return
    # Compute the onset strength
    hop_length_tempo = 512 # 256
    oenv = librosa.onset.onset_strength(
        y=signal, sr=sampling_rate, hop_length=hop_length_tempo
    )
    # Compute the tempogram
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv,
        sr=sampling_rate,
        hop_length=hop_length_tempo,
    )

    algo = rpt.KernelCPD(kernel="rbf").fit(tempogram.T)
    bkps = algo.predict(pen=45)
    bkps_times = librosa.frames_to_time(bkps, sr=sampling_rate, hop_length=hop_length_tempo)
    bkps_time_indexes = (sampling_rate * bkps_times).astype(int).tolist()

    for segment_number, (start, end) in enumerate(
        rpt.utils.pairwise([0] + bkps_time_indexes), start=1
    ):
        segment = signal[start:end]
        #print(f"Segment n°{segment_number} (duration: {segment.size/sampling_rate:.2f} s)")
        ensure_directory_exists(output_audio_dir)
        sf.write(f'{output_audio_dir}/{audio_file_name[:-4]}_segment' + str(segment_number) + '.wav', segment, sampling_rate, 'PCM_16')
        
        # Generate and save Mel spectrogram for the segment
        ensure_directory_exists(output_image_dir)
        S = librosa.feature.melspectrogram(y=segment, sr=sampling_rate)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                 sr=sampling_rate, hop_length=hop_length_tempo,
                                 y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram of Segment {segment_number}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_image_dir, f'{audio_file_name[:-4]}_segment{segment_number}.png'))
        plt.close()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find classe name for target csv.")
    parser.add_argument('-c', '--classes', nargs='+', type=str, help='Space-separated target class names')
    parser.add_argument('-r', '--read', nargs='+', type=str, default='vs', help='directory name to read') # vs, MMSE
    parser.add_argument('-s', '--single_label', type=bool, default=False, help="For taking samples with the single label")
    # parser.add_argument('-m', '--merged_class_name', type=str, help='Destination class name')

    args = parser.parse_args()


    if args.classes == None:
        warnings.warn("Not implemented yet. Please input target class names", UserWarning)
    else:
        TARGET_CLASS_DISPLAY_NAMES = args.classes
        DIR_TO_READ = args.read # Default is 'vs' (vocal separation)
        IS_SINGLE_LABEL = args.single_label
        idx = 0
        for class_display_name in TARGET_CLASS_DISPLAY_NAMES:
            if class_display_name == None:
                warnings.warn("AudioSet_preprocessing_chage_point.py::: class_display_name is None", UserWarning)
                idx = idx+1
                continue
            else:
                # Read audio files
                # TODO Set your audio directory path (YOUR_AUDIO_DIR)
                # As long as you did not change the directory name after the vocal separation, you don't need to manually set DIR_TO_READ
                YOUR_AUDIO_DIR = 'YOUR_AUDIO_DIR'
                audio_dir = os.path.join(YOUR_AUDIO_DIR, f'audio_{DIR_TO_READ}', class_display_name)
                audio_files = os.listdir(audio_dir)

                # Set paths
                output_audio_dir = os.path.join(YOUR_AUDIO_DIR, f'audio_{DIR_TO_READ}_changepoint', class_display_name)
                output_image_dir = os.path.join(YOUR_AUDIO_DIR, f'mel_{DIR_TO_READ}_changepoint', class_display_name)

                for audio_file_name in tqdm(audio_files, desc="Processing audio files"):
                    segment_by_change_point(audio_dir, audio_file_name, output_audio_dir, output_image_dir)

            idx = idx+1