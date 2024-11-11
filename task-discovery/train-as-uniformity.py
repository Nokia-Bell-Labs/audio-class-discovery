import argparse
import os
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import WandbLogger
import sys

from datautils import MyCIFAR10DataModule
# from datamodule.esc50_datamodule import ESC50DataModule
from datamodule.audioset_datamodule import AudiosetDataModule
from models.as_uniformity import ASUniformityTraining
import utils

from datetime import datetime

import time

TMP_PATH = '/tmp/exps/' #'/mnt/wandb/tmp/exps/'

if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                            help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)  # seed for random label
    # wandb parameters
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--group', type=str, default='as-uniformity')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--tags', type=str, nargs='*', default=[])
    parser.add_argument('--nologger', action='store_true', default=False)    
    parser.add_argument('--resume_id', default=os.environ.get('JOB_UUID', datetime.now().strftime("%y%m%d%H%M")))
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--no_resume', dest='resume', action='store_false', default=True)
    parser.add_argument('--entity', type=str, default='task-discovery')
    parser.add_argument('--project_name', type=str, default='task-discovery-repo')
    # train or test
    parser.add_argument('--test', action='store_true', default=False)
    # train parameters
    parser.add_argument('--meta_steps', type=int, default=int(1e5))
    parser.add_argument('--encoder_learning_rate', type=float, default=1e-3)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--save_step_frequency', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--noise', type=float, default=None)

    # Set the feat_type: 
    # (1) original (mel-spectrogram w/o SoundCollage)
    # (2) _vs (mel-spectrogram after SoundCollage's vocal separation)
    # (3) _vs_changepoint (mel-spectrogram after SoundCollage's vocal separation and segmentation)
    parser.add_argument('--feat_type', type=str, default='original')
    # Set the number of data to use
    parser.add_argument('--num_data', type=int, default=-1)
    # all, fore: foreground sounds only (Comp#1), background sounds only (Comp#2)
    parser.add_argument('--vs', type=str, default='all')


    parser = ASUniformityTraining.add_model_specific_args(parser)   # return parser
    parser = pl.Trainer.add_argparse_args(parser)
    # parser = MyCIFAR10DataModule.add_argparse_args(parser)
    
    if '--dataset' in sys.argv:
        dataset_index = sys.argv.index('--dataset')
        # Check the dataset value
        if dataset_index + 1 < len(sys.argv):
            dataset_value = sys.argv[dataset_index + 1]
            # Load DataModule
            if 'cifar10' in dataset_value:
                parser = MyCIFAR10DataModule.add_argparse_args(parser)
            # elif 'esc50' in dataset_value:
            #     parser = ESC50DataModule.add_argparse_args(parser)
            elif 'audioset' in dataset_value:
                parser = AudiosetDataModule.add_argparse_args(parser)
            else:
                raise ValueError(f"Invalid dataset argument: {args.dataset}")

    parser.set_defaults(num_sanity_val_steps=1)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    utils.set_seeds(args.seed)

    SAVE_DIR = args.save_dir if not args.tmp else TMP_PATH
    
    # Set wandb logger
    if not args.nologger:
        # name of the experiment run that will be logged to wanDB
        name = ('tmp-' if args.tmp else '') + args.name.format(**vars(args))
        logger = WandbLogger(
            name=name,
            project=args.project_name,
            entity=args.entity,
            save_dir=SAVE_DIR,
            tags=['as', 'uniformity'] + args.tags,
            group=args.group,
            notes=args.notes,
            id=args.resume_id,
        )
        run = logger.experiment
        print(f'{run.resumed=}')
        checkpoint_callbacks = [
            utils.CheckpointEveryNSteps(
                save_step_frequency=args.save_step_frequency,
            )
        ]
    else:
        logger = None
        checkpoint_callbacks = None

    # Check if there is checkpoint from the previous run
    ckpt_path = os.path.join(SAVE_DIR, args.project_name, args.resume_id, 'checkpoints', 'checkpoint.ckpt')
    
    if not os.path.exists(ckpt_path):
        if logger is not None and run.resumed:
            print(f'====> FAILED to find a checkpoint from the previous run: {ckpt_path}')
        ckpt_path = None

    if not args.resume:
        ckpt_path = None
    
    # Load checkpoint
    if args.ckpt and ckpt_path is None:
        model = ASUniformityTraining.load_from_checkpoint(args.ckpt, **vars(args))
        print(f'====> Loaded from checkpoint: {args.ckpt}')
    else:
        model = ASUniformityTraining(**vars(args))

    trainer = pl.Trainer(
        gpus= [1], # torch.cuda.device_count()
        logger=logger,
        log_every_n_steps=1,
        callbacks=checkpoint_callbacks,
        max_steps=args.meta_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        resume_from_checkpoint=ckpt_path,
    )
    
    if not args.test:
        trainer.fit(model, ckpt_path=ckpt_path)
        trainer.test(model)
    else:
        trainer.test(model, ckpt_path=ckpt_path)

