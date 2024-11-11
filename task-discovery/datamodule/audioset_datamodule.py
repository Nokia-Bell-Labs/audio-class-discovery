'''© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pytorch_lightning as pl
from typing import Any, Optional, Union, List
import argparse
import random
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import glob
from tqdm import tqdm
from torchvision import datasets, transforms
import shutil

import warnings

import pandas as pd
from collections import Counter

import sys

# TODO Set to the appropriate path to audioset dataset
AUDIOSET_DATASET_DIR = 'YOUR_AUDIO_DIR'
# CLASS_NAME_TO_TASK_ID: {'original':0, 'pop_music': 1, 'classical_music': 2, 'outside_urban_or_manmade': 3}

class MyAudiosetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # TODO Set self.CLASS_NAME to test, self.FEATURE_TYPE, and self.VS
        self.CLASS_NAME = 'CLASS_NAME'
        self.FEATURE_TYPE = 'FEATURE_TYPE' # This should be either 'original' or "feat_type" argument value you used when you ran train-as-uniformity.py
        self.VS = 'VS'                     # This should be "vs" argument value you used when you ran train-as-uniformity.py
        
        csv_file = os.join(AUDIOSET_DATASET_DIR, 'meta', f'{self.CLASS_NAME}{self.FEATURE_TYPE}.csv') # YTID, start_seconds, end_seconds, positvie_labels
        if 'highqual' in self.CLASS_NAME:
            # column_names = ['filename', 'start_seconds', 'end_seconds', 'labels']
            metadata_df = pd.read_csv(csv_file)
            if df.shape[1] > 4: # When there is 'final_label' column
                metadata_df = metadata_df.dropna(subset=['final_label'])
                metadata_df = metadata_df.drop(columns=['labels'])
                metadata_df = metadata_df.rename(columns={'final_label': 'labels'})
        else:
            metadata_df = pd.read_csv(csv_file)  
            
            if metadata_df.columns[0] != 'filename':
                # dataframes where the first row is set as column titles
                new_row = pd.DataFrame([metadata_df.columns], columns=metadata_df.columns)
                metadata_df = pd.concat([new_row, metadata_df]).reset_index(drop=True)
            
            metadata_df.columns = ['filename', 'start_seconds', 'end_seconds', 'labels']
                
        self.metadata = metadata_df

    def __len__(self):
        return len(self.metadata)

    def __getitem__ (self, index):
        # filename format is {file_id}_{start_seconds}.png
        
        if len(self.FEATURE_TYPE) != 0:
            # filename in meta csv file for processed audio is the exact file name without the file extension.
            image_file = os.path.join(AUDIOSET_DATASET_DIR, f"mel_{self.FEATURE_TYPE}", self.CLASS_NAME,
                                    self.metadata.iloc[index, 0] + '.png')
        else:
            # No processing case.
            # filename in meta csv file for original audio is file ID.
            image_file = os.path.join(AUDIOSET_DATASET_DIR, f"mel", self.CLASS_NAME,
                                    self.metadata.iloc[index, 0] + '_' + self.metadata.iloc[index, 1] + '.png')
        image = Image.open(image_file)

        # Adjust to the input size of Encoder
        resize_transform = transforms.Resize((64, 64))
        image = resize_transform(image)

        # Convert image to a tensor (C x H x W)
        image = transforms.ToTensor()(image)


        def find_class_name_by_class_id(TARGET_CLASS_IDS):
            csv_file = os.join(AUDIOSET_DATASET_DIR, 'meta', 'class_labels_indices.csv')
            class_labels_indicies_df = pd.read_csv(csv_file)

            # dictionary mapping {mid: display_name}
            mid_to_display_name = dict(zip(class_labels_indicies_df['mid'], class_labels_indicies_df['display_name']))

            class_names = []
            for item in TARGET_CLASS_IDS:
                if item in mid_to_display_name:
                    class_names.append(mid_to_display_name[item].lower().replace(' ', '_')).replace('(','').replace(')','').replace(',','')
                else:
                    # If the item is not found in the mid_to_display_name dictionary, show warning
                    warnings.warn(f"{item} not found in class_labels_indices.csv.", UserWarning)

            return class_names

        # Get class labeles (targets) from the metadata
        targets_id = self.metadata.iloc[index, 3]
        targets_id = targets_id.split(',')
        targets = find_class_name_by_class_id(targets_id)
        if targets == []:
            warnings.warn(f"__getitem__::: no corresponding targets.", UserWarning)
            sys.exit()

        # Apply the transformation, if available
        if self.transform is not None:
            image = self.transform(image)

        #return image, targets
        return image, targets[0] # Return the first label when there are multiple labels


class MyAudiosetTrainDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        return *super().__getitem__(index), index

class MyAudiosetValDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        return *super().__getitem__(index), index

class MyAudiosetTestDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        return *super().__getitem__(index), index

class AudiosetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Optional[str] = AUDIOSET_DATASET_DIR,
        num_workers: int = 32,
        batch_size: int = 64,
        test_batch_size: Optional[int] = None,
        data_seed: int = 42,    
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = True,
        task_type: str = 'real',
        random_labelling: bool = False,
        random_labelling_seed: Optional[int] = None,
        n_classes: int = 2, #
        persistent_workers: bool = False,
        return_indicies: bool = False,
        image_size: int = 64,
        task_idx: int=-1,
        num_epoch: int=10,
        test_proportion: float=0.1,
        val_proportion: float=0.1,
        # speech, pop_music, classical_music, outside_urban_or_manmade, outside_rural_or_natural, domestic_sounds_home_sounds
        selective: str = 'original',
        feat_type: str = '',
        gt2class: Optional[str] = None,
        num_data: int = -1,
        vs: str = 'all',
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = data_seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.return_indicies = return_indicies
        self.pin_memory = pin_memory
        self.dims = (3, image_size, image_size)

        self.random_labelling = random_labelling
        self.random_labelling_seed = random_labelling_seed if random_labelling_seed is not None else self.seed
        self.task_type = task_type
        print(
            f'[AudioSetDatamodule] ===> : Shuffle={shuffle}, Data_seed={data_seed}, Persistent_workers={persistent_workers}, Drop_last={drop_last}')
        self._num_classes = n_classes

        self.task_idx = task_idx #self.task_idx = CLASS_NAME_TO_TASK_ID[selective]
        self.num_epoch = num_epoch
        self.test_proportion = test_proportion
        self.val_proportion = val_proportion

        self.test_batch_size = test_batch_size or self.batch_size
        self.persistent_workers = persistent_workers

        self.selective = selective
        self.feat_type = feat_type
        self.num_data = num_data
        self.vs = vs
        
        arg_name = kwargs['name']   # {dataset}-{timestamp}
        parts = arg_name.split('-')
        self.class_name = parts[0].replace('audioset', '')
        self.timestamp = parts[1]
        
        # Subdirectory path
        train_dir_path = os.path.join(self.dataset_path, f"train_{self.class_name}_{self.feat_type}_{self.batch_size}_{self.num_epoch}_{self.timestamp}")
        test_dir_path = os.path.join(self.dataset_path, f"test_{self.class_name}_{self.feat_type}_{self.batch_size}_{self.num_epoch}_{self.timestamp}")
        val_dir_path = os.path.join(self.dataset_path, f"val_{self.class_name}_{self.feat_type}_{self.batch_size}_{self.num_epoch}_{self.timestamp}")

        self._gt2class = gt2class #None
        if isinstance(gt2class, str) and gt2class != '' and not self.random_labelling:
            if self._gt2class == 'check_as':
                print(f"{self._gt2class=}")
            else:
                self._gt2class = {gt: i for i, clss in enumerate(gt2class.split('|')) for gt in clss.split(',')}
                print(f"{self._gt2class=}") # 'trainOnDataLabeles'


        self.targets = []


        def empty_or_refresh(dir_path):
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            else:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)

        def split_dataset():
            # Clean subdirectoreis for training, testing, and validation
            empty_or_refresh(train_dir_path)
            empty_or_refresh(test_dir_path)
            empty_or_refresh(val_dir_path)
            
            # Get the list of image files in the data directory
            print(
                f'[AudiosetDatamodule] ===> dataset_path is {dataset_path}')
            if feat_type == 'original':
                all_files = os.listdir(f'{dataset_path}/mel/{self.class_name}')
            else: # For specific feature types
                all_files = os.listdir(f'{dataset_path}/mel{feat_type}/{self.class_name}')
            
            img_files = [f for f in all_files if f.endswith('.png')]
            
            def find_labels(filename, start_seconds, labels_df):
                matched_row = labels_df[(labels_df['filename'] == filename) & (labels_df['start_seconds'] == int(start_seconds))]
                if matched_row.empty:
                    print(f"Warning: {filename} not found in meta csv file.")
                    return False
                labels = matched_row.iloc[0]['labels'].split(',')
                
                return labels[0]
            
            # Set labels
            print(f"{self._gt2class=}")
            if self._gt2class is not None:
                # Real labels
                if self.feat_type == 'original':
                    if self.selective == 'original':
                        meta_file_path = os.path.join(AUDIOSET_DATASET_DIR, 'meta', f'{self.class_name}.csv')
                    else:
                        meta_file_path = os.path.join(AUDIOSET_DATASET_DIR, 'meta', f'{self.class_name}_{self.selective}.csv')
                else:
                    if self.selective == 'original':
                        meta_file_path = os.path.join(AUDIOSET_DATASET_DIR, 'meta', f'{self.class_name}{self.feat_type}.csv')
                    else:
                        meta_file_path = os.path.join(AUDIOSET_DATASET_DIR, 'meta', f'{self.class_name}{self.feat_type}_{self.selective}.csv')
                
                with open(meta_file_path, 'r') as meta_csv:
                    meta_df_column_names = ['filename', 'start_seconds', 'end_seconds', 'labels'] # Discard YTID, start_seconds, end_seconds
                    meta_df = pd.read_csv(meta_csv, header=None, names=meta_df_column_names)
                
                new_target = []
                
                print(f"[AudiosetDatamodule] ===> {self.task_type}")
                if self.task_type == 'random':
                    img_file_num = len(img_files)
                    random_targets = [random.random() for _ in range(img_file_num)]
                    self.targets = random_targets
                    new_target = self.targets
                    
                    self.num_classes = 2 # Always random binary labeling. Replacable with len(list(set(new_target)))

                else:
                    # Find real labels
                    for img_file in img_files:
                        if self.feat_type == 'original':
                            img_file_name_splitted = img_file.rsplit('_', 1)
                            file_id = img_file_name_splitted[0]
                            start_seconds = img_file_name_splitted[1].replace('.png', '')
                            
                            target = find_labels(file_id, start_seconds, meta_df)    # meta csv for original samples has file ID as 'filename' (id) column value
                        else:
                            num_postfix = self.feat_type.count('_')
                            img_file_name_splitted = img_file.rsplit('_', num_postfix)
                            file_id = img_file_name_splitted[0]
                            start_seconds = img_file_name_splitted[1].replace('.png', '')
                            
                            img_file_name = img_file.replace('.png', '')    # meta csv for preprocessed samples has file NAME without file extension as 'filename' (id) column value
                            target = find_labels(img_file_name, start_seconds, meta_df)
                        
                        new_target.append(target)
                    
                    # Get unique labels
                    unique_labels = list(set(new_target))
                    unique_labels.sort()
                    print(f"[AudiosetDatamodule] ===> {unique_labels=}")

                    # Create a dictionary mapping unique values to integers
                    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
                    
                    # Replace labels in the new_target with their corresponding integer values
                    self.targets = [label_to_int[label] for label in new_target]
                    
                    if self.task_idx > -1:   # Check AS
                        # Make the task as binary [Target task (task_idx) or Not]. e.g., female speech vs. Not female speech
                        new_target_binary = [1 if item == self.task_idx else 0 for item in self.targets]
                        new_target = new_target_binary
                        self.targets = new_target_binary
                        unique_labels = list(set(new_target_binary))
                        unique_labels.sort()
                    elif self.task_idx == -2:   # Check AS with random label
                        g = torch.Generator().manual_seed(self.random_labelling_seed)
                        new_target = torch.randint(0, 2, (len(img_files),), generator=g).tolist()
                        self.targets = new_target
                        print(f"[AudiosetDatamodule] ===> RANDOM TASK")
                    
                
            else: #if self.random_labelling_seed != self.seed:  #if self.random_labelling:
                # Random labels
                g = torch.Generator().manual_seed(self.random_labelling_seed)
                new_target = torch.randint(0, 2, (len(img_files),), generator=g).tolist()
                self.targets = new_target

            # Split to tain, val, test
            img_with_labels = list(zip(img_files, new_target)) # Combine img_files and labels using zip
            random.shuffle(img_with_labels)

            # Take foreground (Comp#1) or background (Comp#2) sound if 'vs' is not 'all'.
            print(f"{self.vs=}")
            if self.vs == 'fore':
                filtered_img_with_labels = [item for item in img_with_labels if '_fore' in item[0]]
                img_with_labels = filtered_img_with_labels
            elif self.vs == 'back':
                filtered_img_with_labels = [item for item in img_with_labels if '_back' in item[0]]
                img_with_labels = filtered_img_with_labels

            # For specific number of data to use
            print(f"{self.num_data=}")
            if self.num_data == -1:
                random.shuffle(img_with_labels)
                num_imgs = len(img_with_labels)
            else:
                num_imgs = self.num_data

            num_test = int(self.test_proportion * num_imgs)
            num_val = int(self.val_proportion * num_imgs)
            num_train = num_imgs - num_test - num_val

            train_img_with_labels = img_with_labels[:num_train]
            test_img_with_labels = img_with_labels[num_train:num_train+num_test]
            val_img_with_labels = img_with_labels[num_train+num_test:]

            # Unzip the lists to separate img_files and labels
            train_data, train_targets = zip(*train_img_with_labels)
            test_data, test_targets = zip(*test_img_with_labels)
            val_data, val_targets = zip(*val_img_with_labels)
            
            print(f'[AudiosetDatamodule] ===> (train#, test#, val#) == ({num_train}, {num_test}, {num_val})')
            
            # Create necessary subdirectories
            unique_targets = set(self.targets)  # Get unique targets
            for item in unique_targets:
                os.makedirs(os.path.join(train_dir_path, str(item)))
                os.makedirs(os.path.join(test_dir_path, str(item)))
                os.makedirs(os.path.join(val_dir_path, str(item)))

            if self.feat_type == 'original':
                for img_file, target in zip(train_data, train_targets):
                    source_path = os.path.join(dataset_path, 'mel', self.selective, img_file)
                    destination_path = os.path.join(train_dir_path, str(target), img_file)
                    shutil.copyfile(source_path, destination_path)
                for img_file, target in zip(test_data, test_targets):
                    source_path = os.path.join(dataset_path, 'mel', self.selective, img_file)
                    destination_path = os.path.join(test_dir_path, str(target), img_file)
                    shutil.copyfile(source_path, destination_path)
                for img_file, target in zip(val_data, val_targets):
                    source_path = os.path.join(dataset_path, 'mel', self.selective, img_file)
                    destination_path = os.path.join(val_dir_path, str(target), img_file)
                    shutil.copyfile(source_path, destination_path)
            else:
                for img_file, target in zip(train_data, train_targets):
                    source_path = os.path.join(dataset_path, f'mel{self.feat_type}', self.selective, img_file)
                    destination_path = os.path.join(train_dir_path, str(target), img_file)
                    shutil.copyfile(source_path, destination_path)
                for img_file, target in zip(test_data, test_targets):
                    source_path = os.path.join(dataset_path, f'mel{self.feat_type}', self.selective, img_file)
                    destination_path = os.path.join(test_dir_path, str(target), img_file)
                    shutil.copyfile(source_path, destination_path)
                for img_file, target in zip(val_data, val_targets):
                    source_path = os.path.join(dataset_path, f'mel{self.feat_type}', self.selective, img_file)
                    destination_path = os.path.join(val_dir_path, str(target), img_file)
                    shutil.copyfile(source_path, destination_path)
            

        split_dataset()
        dataset_train_cls = MyAudiosetTrainDataset if return_indicies else datasets.ImageFolder
        dataset_test_cls = MyAudiosetTestDataset if return_indicies else datasets.ImageFolder
        dataset_val_cls = MyAudiosetValDataset if return_indicies else datasets.ImageFolder
        
        
        self.dataset_train = dataset_train_cls(
            train_dir_path,
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        )
        self.dataset_test = dataset_test_cls(
            test_dir_path,
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        )
        self.dataset_val = dataset_val_cls(
            val_dir_path,
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        )
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--data_seed', type=int, default=42)
        parser.add_argument('--random_labelling_seed', type=int, default=42)
        parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
        parser.add_argument('--shuffle', dest='shuffle', action='store_true')
        # parser.set_defaults(shuffle=True)
        parser.set_defaults(shuffle=False)
        # parser.add_argument('--n_classes', type=int, default=2)
        parser.add_argument('--no_drop_last', dest='drop_last', action='store_false', default=True)
        parser.add_argument('--return_indicies', action='store_true', default=False)
        parser.add_argument('--persistent_workers', action='store_true', default=False)
        parser.add_argument('--dataset_path', type=str, default='')
        return parser

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup()

    def _data_loader(
            self,
            dataset: torch.utils.data.Dataset,
            generator: Any = None,
            shuffle: bool = False,
            persistent_workers: bool = False,
            batch_size: int = None,
            drop_last: bool = None,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=self.num_workers,
            drop_last=self.drop_last if drop_last is None else drop_last,
            pin_memory=self.pin_memory,
            worker_init_fn=AudiosetDataModule._worker_init_fn,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(
            self,
            generator: Optional[torch.Generator] = None,
            persistent_workers: bool = False,
            batch_size: int = None,
    ) -> torch.utils.data.DataLoader:
        """ The train dataloader """
        persistent_workers = persistent_workers or self.persistent_workers
        return self._data_loader(self.dataset_train, shuffle=self.shuffle, generator=generator,
                                 persistent_workers=persistent_workers, batch_size=batch_size)

    def val_dataloader(self, persistent_workers: bool = False, batch_size: int = None) -> torch.utils.data.DataLoader:
        """ The val dataloader """
        persistent_workers = persistent_workers or self.persistent_workers
        batch_size = batch_size or self.test_batch_size
        return self._data_loader(self.dataset_val, persistent_workers=persistent_workers, batch_size=batch_size,
                                 drop_last=False)   # drop_last=True)#

    def test_dataloader(self, persistent_workers: bool = False, batch_size: int = None) -> torch.utils.data.DataLoader:
        """ The train dataloader """
        batch_size = batch_size or self.test_batch_size
        return self._data_loader(self.dataset_val, persistent_workers=persistent_workers, batch_size=batch_size,
                                 drop_last=False)   # drop_last=True)#

    @staticmethod
    def _worker_init_fn(_id):
        seed = torch.utils.data.get_worker_info().seed % 2 ** 32
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)