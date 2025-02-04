'''© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear '''
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

import matplotlib.pyplot as plt
from IPython.display import Audio
import scipy
from scipy.io import wavfile

import argparse
import os
import warnings
from tqdm import tqdm

import pandas as pd


def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find classe name for target csv.")
    parser.add_argument('-i', '--id', type=str, help='Target WanDB repo ID')
    parser.add_argument('-c', '--classname', default= 'audiosetspeech', type=str, help='Target class name')
    parser.add_argument('-t', '--top', type=int, default=32, help='Number of classes you want to check')

    args = parser.parse_args()

    if args.id == None:
        warnings.warn("Not implemented yet. Please input target class names", UserWarning)

    else:
        TARGET_WANDB_ID = args.id
        TARGET_CLASS = args.classname
        TOP_N = args.top

        # Load the yamnet model
        model = hub.load('https://tfhub.dev/google/yamnet/1')

        # Find the name of the class with the top score when mean-aggregated across frames.
        class_map_path = model.class_map_path().numpy()
        class_names = class_names_from_csv(class_map_path)

        idx = 0

        if TARGET_WANDB_ID == None:
            warnings.warn("sound_classification_discovered_tasks.py::: class_display_name is None", UserWarning)
        else:
            for task_id in range(32):   # 0 to 31
                for class_id in range(2):   # 0 to 1
                    # Read audio files
                    # TODO: Set your audio dir path. If you did not change path when retreiving top N samples, you may not need to edit the path below
                    audio_dir = os.path.join('../task-discovery/retrieve_top-k_samples',
                                             str(TARGET_CLASS), f'audio_samples_{TARGET_WANDB_ID}', f'taskid_{task_id}',
                                             str(class_id))
                    audio_files = os.listdir(audio_dir) # name only

                    print(f"Number of audio files to classify using YAMNet is {len(audio_files)}")

                    # Set output csv paths
                    output_class_dir = os.path.join('../task-discovery/retrieve_top-k_samples',
                                             str(TARGET_CLASS), f'yamnet_validation_{TARGET_WANDB_ID}', f'taskid_{task_id}',
                                             str(class_id))
                    ensure_directory_exists(output_class_dir)

                    column_names = ['filename', 'y_indices', 'y_display_name', 'y_scores']


                    df = pd.DataFrame(columns=column_names)

                    for audio_file_name in tqdm(audio_files, desc="Classifying audio files"):
                        sample_rate, wav_data = wavfile.read(audio_dir + '/' + audio_file_name, 'rb')
                        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

                        # wav_data needs to be normalized to values in [-1.0, 1.0]
                        waveform = wav_data / tf.int16.max
                        waveform = waveform.reshape(-1)

                        # Run the model, check the output
                        scores, embeddings, spectrogram = model(waveform)
                        scores_np = scores.numpy()
                        spectrogram_np = spectrogram.numpy()
                        infered_class = class_names[scores_np.mean(axis=0).argmax()]

                        mean_scores = np.mean(scores, axis=0)
                        top_class_indices = np.argsort(mean_scores)[::-1][:TOP_N]
                        top_class_display_names = []
                        top_class_scores = []
                        for i in range(0, TOP_N, 1):
                            top_class_display_names.append(class_names[top_class_indices[i]])
                            top_class_scores.append(mean_scores[top_class_indices[i]])   #top_class_scores.append(np.argsort(mean_scores)[::-1][i])

                        new_row = {'filename': audio_file_name,
                                   'y_indices': top_class_indices,
                                   'y_display_name': top_class_display_names,
                                   'y_scores': top_class_scores}
                        new_row = pd.DataFrame(new_row)
                        df = pd.concat([df, new_row], axis=0, ignore_index=True)

                    df.to_csv(output_class_dir + '/val.csv', index=False)


