'''© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear '''

import os
import argparse

import numpy as np
from tqdm import tqdm

import librosa
import librosa.display
import soundfile as sf

import matplotlib.pyplot as plt

from librosa.util.exceptions import ParameterError
from scipy.interpolate import interp1d

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# def compute_mfcc(audio, sr):
#     hop_length = int(0.01 * sr) # 10 ms overlap
#     win_length = int(0.025 * sr)    # 25 ms window
#     n_mels = 64 # 64 mel frequency bands
#     n_mfcc = 64 # extract 64 mfcc coefficients
    
#     # Compute STFT and convert to power spectrogram
#     S = librosa.stft(audio, n_fft=win_length, hop_length=hop_length, win_length=win_length)
#     S_power = np.abs(S) ** 2  # Compute power spectrogram

#     # Apply Mel filter banks
#     mel_spec = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=n_mels)

#     # Convert to log scale (log-mel spectrogram)
#     mel_log = librosa.power_to_db(mel_spec, ref=np.max)

#     # Compute MFCCs from the log-mel spectrogram
#     mfcc = librosa.feature.mfcc(S=mel_log, sr=sr, n_mfcc=n_mfcc)

#     # Resample MFCC frames to 64 time steps using linear interpolation
#     num_frames = mfcc.shape[1]
#     target_frames = 64  # Fixed number of time steps

#     if num_frames != target_frames:
#         x_old = np.linspace(0, 1, num_frames)
#         x_new = np.linspace(0, 1, target_frames)
#         mfcc_resampled = np.array([interp1d(x_old, mfcc[i, :], kind='linear')(x_new) for i in range(n_mfcc)])
#     else:
#         mfcc_resampled = mfcc

#     return mfcc_resampled
    
# def save_mfcc_image(mfcc, image_path):
#     """Save the MFCC as an image."""
#     plt.figure(figsize=(10, 10))
#     librosa.display.specshow(mfcc, x_axis='time', y_axis='mel', cmap='viridis')
#     plt.axis('off')
#     plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300)
#     plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find classe name for target csv.")
    parser.add_argument('-c', '--classes', nargs='+', type=str, help='Space-separated target class names')

    args = parser.parse_args()


    if args.classes == None:
        warnings.warn("Not implemented yet. Please input target class names", UserWarning)
    else:
        TARGET_CLASS_DISPLAY_NAMES = args.classes
        
        for class_display_name in TARGET_CLASS_DISPLAY_NAMES:
            if class_display_name == None:
                warnings.warn("AudioSet_preprocessing_speech.py::: class_display_name is None", UserWarning)
                continue
            
            else:
                # Read audio files
                # TODO Set your audio directory path
                YOUR_AUDIO_DIR = 'YOUR_AUDIO_DIR'
                audio_dir = os.path.join(YOUR_AUDIO_DIR, class_display_name) # "/mnt/audioset/audio"
                audio_files = os.listdir(audio_dir)
                if class_display_name == 'speech':
                    audio_files = audio_files[:10000]

                # Set paths
                output_audio_dir = os.path.join(YOUR_AUDIO_DIR, 'audio_vs', class_display_name)
                output_image_dir = os.path.join(YOUR_AUDIO_DIR, 'mfcc_vs', class_display_name)
                ensure_directory_exists(output_audio_dir)
                ensure_directory_exists(output_image_dir)
                
                for audio_file_name in tqdm(audio_files, desc="Processing audio files"):
                    audio_signal, sr = librosa.load(audio_dir + '/' + audio_file_name)
                    
                    # audio_signal = pad_audio_to_10_seconds(audio_signal, sr)
                    
                    # Compute the spectrogram magnitude and phase
                    S_full, phase = librosa.magphase(librosa.stft(audio_signal))
                    
                    # Compare frames using cosine similarity and aggregate similar frames by taking their (per-frequency) median value
                    # Constrain similar frames to be separated by at least 2 seconds to avoid being biased by local continuity
                    valid_width = min(librosa.time_to_frames(2, sr=sr), (S_full.shape[1] - 1) // 2)
                    if valid_width != librosa.time_to_frames(2, sr=sr): # Exp case: some samples are shorter than 2 seconds -> Pass the sample
                        print(f"{valid_width=}")
                        continue
                    try:
                        S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(valid_width))
                    except ParameterError as e:
                        continue

                    # Output shouldn't be greater than the input
                    S_filter = np.minimum(S_full, S_filter)
                    
                    # Marging to reduce bleed between the vocals and instrumentation masks
                    # Interpretation: When separating the instrumental part (margin_i), the algorithm will be less aggressive, allowing for some vocal components to remain in the instrumental separation
                    margin_i, margin_v = 2, 10
                    power = 2
                    
                    mask_i = librosa.util.softmask(S_filter,
                                                   margin_i * (S_full - S_filter),
                                                   power=power)
                    mask_v = librosa.util.softmask(S_full - S_filter,
                                                   margin_v * S_filter,
                                                   power=power)
                    
                    # Simply multiply the masks with the input spectrum to separate the components
                    S_foreground = mask_v * S_full
                    S_background = mask_i * S_full
                    
                    # Convers S_foreground and S_background back to audio signals
                    y_foreground = librosa.istft(S_foreground * phase)
                    y_background = librosa.istft(S_background * phase)
                    
                    sf.write(os.path.join(output_audio_dir, f"{audio_file_name[:-4]}_fore.wav"), y_foreground, sr, 'PCM_16')
                    sf.write(os.path.join(output_audio_dir, f"{audio_file_name[:-4]}_back.wav"), y_background, sr, 'PCM_16')
                    
                    
                    # Compute and save MFCC images instead of Mel spectrograms
                    # mfcc_foreground = compute_mfcc(y_foreground, sr)
                    # mfcc_background = compute_mfcc(y_background, sr)

                    # save_mfcc_image(mfcc_foreground, os.path.join(output_image_dir, f"{audio_file_name[:-4]}_fore.png"))
                    # save_mfcc_image(mfcc_background, os.path.join(output_image_dir, f"{audio_file_name[:-4]}_back.png"))