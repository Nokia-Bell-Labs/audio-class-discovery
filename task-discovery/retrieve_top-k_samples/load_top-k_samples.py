import os
import shutil
import pandas as pd

# TODO: Set the task ID (wandb id), checkpoint name, repo name, dataset name
DATASET = 'your_dataset_name'
TASK_ID = 'your_task_id'

def ensure_directory_exists(directcdory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def copy_img_to_dest(csv_file_path, destination_directory):
    # Read CSV files
    df = pd.read_csv(csv_file_path)

    # Convert filenames to absolute file paths
    absolute_file_paths = [os.path.join(destination_directory, filename) for filename in df['filename']]

    # Create the destination directory if it does not exist
    os.makedirs(destination_directory, exist_ok=True)

    # Copy files to the destination directory
    for file_path in absolute_file_paths:
        shutil.copy(file_path, destination_directory)

def copy_audio_to_dest(csv_file_path, destination_directory):
    # Read CSV files
    df = pd.read_csv(csv_file_path)

    def extract_filename(file_path):
        # Extract the filename from the file path
        filename = os.path.basename(file_path)

        # Replace the file extension '.png' with '.wav'
        filename = os.path.splitext(filename)[0] + '.wav'


        return filename

    # Create the destination directory if it does not exist
    os.makedirs(destination_directory, exist_ok=True)
    os.makedirs(f"{DATASET}/audio_speech_samples_{TASK_ID}/taskid_{task_id}/{class_id}", exist_ok=True)
    os.makedirs(f"{DATASET}/image_speech_samples_{TASK_ID}/taskid_{task_id}/{class_id}", exist_ok=True)
 
    # TODO: Set to the appropriate path to audioset dataset
    AUDIOSET_DATASET_DIR = 'YOUR_AUDIO_DIR'

    for filename in df['filename']:
        audio_filename = extract_filename(filename)
        audio_filepath = os.path.join(AUDIOSET_DATASET_DIR, 'audio_vs_changepoint', DATASET[8:], audio_filename)
        mfcc_filepath = os.path.join(AUDIOSET_DATASET_DIR, 'mfcc_vs_changepoint', DATASET[8:], audio_filename)

        shutil.copy(audio_filepath, f"{DATASET}/audio_samples_{TASK_ID}/taskid_{task_id}/{class_id}")
        shutil.copy(mfcc_filepath, f"{DATASET}/image_samples_{TASK_ID}/taskid_{task_id}/{class_id}")


if __name__ == "__main__":
    # Set the paths to CSV file and base directory
    for task_id in range(32):
        for class_id in range(2):
            csv_file_path = f"{DATASET}/filenames_{TASK_ID}/taskid_{task_id}_class{class_id}_filenames.csv"
            mfcc_directory = f"{DATASET}/image_samples_{TASK_ID}/taskid_{task_id}/{class_id}"
            audio_directory = f"{DATASET}/audio_samples_{TASK_ID}/taskid_{task_id}/{class_id}"

            ensure_directory_exists(mfcc_directory)
            ensure_directory_exists(audio_directory)

            copy_img_to_dest(csv_file_path, mfcc_directory)
            copy_audio_to_dest(csv_file_path, audio_directory)

