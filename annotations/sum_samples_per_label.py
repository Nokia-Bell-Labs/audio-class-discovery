'''© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear '''
import pandas as pd

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        #warnings.warn("There is no csv file")
        os.makedirs(directory_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find classe name for target csv.")
    parser.add_argument('-i', '--id', type=str, help='Target id')
    parser.add_argument('-c', '--classname', type=str, help='Target dataset name (e.g., audiospeech)')

    args = parser.parse_args()

    if args.classname == None or args.id == None:
        warnings.warn("Missing arguments")
    else:
        TARGET_WANDB_ID = args.id
        TARGET_CLASS = args.classname

        sum_samples_per_task_df = pd.DataFrame(columns=['task_id', 'class', 'label', 'num_sample'])
        output_csv_file_path = f"../task-discovery/retrieve_top-k_samples/{TARGET_CLASS}/sum_samples_{TARGET_WANDB_ID}"
        ensure_directory_exists(output_csv_file_path)
        
        # Set the paths to CSV file and base directory
        for task_id in range(32):
            df_task = pd.DataFrame(columns=['filename', 'y_indices', 'y_display_name', 'y_scores', 'class'])
            for class_id in range(2):
                csv_file_path = f"../task-discovery/retrieve_top-k_samples/{TARGET_CLASS}/yamnet_validation_{TARGET_WANDB_ID}/taskid_{task_id}/{class_id}/val.csv"
                #img_directory = f"{TARGET_CLASS}/image_samples_{TARGET_WANDB_ID}/taskid_{task_id}/{class_id}"
                #audio_directory = f"{TARGET_CLASS}/audio_samples_{TARGET_WANDB_ID}/taskid_{task_id}/{class_id}"

                #ensure_directory_exists(img_directory)
                #ensure_directory_exists(audio_directory)

                df = pd.read_csv(csv_file_path) # [filename, y_indices, y_display_name, y_scores]
                df['class']=class_id
                df_task = pd.concat([df_task, df], ignore_index=True)

            # Extract unique y_display_name
            unique_labels = df_task['y_display_name'].unique()
            #print(f"{unique_labels=}")

            for label in unique_labels:
                df_task_matched_with_label = df_task[df_task['y_display_name'] == label]
                class_counts = df_task_matched_with_label['class'].value_counts()
                class_0_count = class_counts.get(0,0)
                class_1_count = class_counts.get(1, 0)
                sum_samples_per_task_df = sum_samples_per_task_df.append({'task_id': task_id,
                                                        'class': int(0),
                                                        'label': label,
                                                        'num_sample': class_0_count}, ignore_index=True)
                sum_samples_per_task_df = sum_samples_per_task_df.append({'task_id': task_id,
                                                        'class': int(1),
                                                        'label': label,
                                                        'num_sample': class_1_count}, ignore_index=True)

        sum_samples_per_task_df.to_csv(f"{output_csv_file_path}/sum_samples_per_task.csv", index=False)
        sum_samples_df = sum_samples_per_task_df.groupby('label')['num_sample'].sum().reset_index()
        sum_samples_df = sum_samples_df.sort_values(by='num_sample', ascending=False)
        sum_samples_df.to_csv(f"{output_csv_file_path}/sum_samples.csv", index=False)