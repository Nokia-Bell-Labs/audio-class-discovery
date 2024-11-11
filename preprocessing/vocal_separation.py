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

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def mel(audio, sr, image_path):
    # Save Mel-spectrogram
    plt.figure(figsize=(10, 10))

    # Generate the Mel spectrogram
    n_fft = min(2048, len(audio) - 1)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft = n_fft)
    
    # Convert power spectrum to dB scale
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Ensure S_dB is 2D before plotting
    # if S_dB.ndim == 3 and S_dB.shape[2] == 1:
    #     S_dB = S_dB.reshape(S_dB.shape[0], S_dB.shape[1])
    
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    return

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
                output_image_dir = os.path.join(YOUR_AUDIO_DIR, 'mel_vs', class_display_name)
                ensure_directory_exists(output_audio_dir)
                ensure_directory_exists(output_image_dir)
                
                for audio_file_name in tqdm(audio_files, desc="Processing audio files"):
                    audio_signal, sr = librosa.load(audio_dir + '/' + audio_file_name)
                    
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
                    
                    mel(y_foreground, sr, os.path.join(output_image_dir, f"{audio_file_name[:-4]}_fore.png"))
                    mel(y_background, sr, os.path.join(output_image_dir, f"{audio_file_name[:-4]}_back.png"))