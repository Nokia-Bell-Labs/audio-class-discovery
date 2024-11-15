import argparse

import pandas as pd
import pytorch_lightning as pl
pl.Trainer(accelerator='gpu', devices=8)
import os

import torch
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from collections import defaultdict
from functools import partial
from tkinter.messagebox import NO
from typing import Any, List, Optional, Union
from einops import rearrange
from pandas import DataFrame
from sklearn.metrics import adjusted_mutual_info_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from torch.utils.data import DataLoader, TensorDataset

from torch import nn
from models.agreement_score import ClassificationAgreementScore
from models.supervised import TwoSupervisedModels
from models.tasks import CIFARClassificationTask, CIFAREmbeddingClassificationTask, \
    AudiosetClassificationTask, AudiosetEmbeddingClassificationTask
# TO REMOVE: ESC50ClassificationTask, ESC50EmbeddingClassificationTask, \
matplotlib.rcParams["figure.dpi"] = 200

from .td_encoder import TaskDiscoveryEncoder
from .taskness_score import AgreementLoss, FOXASLoss, TrainingProxyASLoss, TrainingXASLoss, entropy_with_logits


from tiny_imagenet import TinyImageNetDataModule
from datautils import REAL_TASKS, MyCIFAR10DataModule
from datamodule.audioset_datamodule import AudiosetDataModule 
import utils
from utils import partialclass

# Define a PyTorch module that computes a contrastive loss based on the uniformity of embedding vectors
class Uniformity(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=2., multiplier=1, distributed=False):
        super().__init__()
        self.tau = tau
        self.distributed = distributed
        assert multiplier == 1
        self.multiplier = 1
        self.norm = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        # The embedding vectors 'z' are normalized using L2 normalization.
        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TaskdiscoveryTODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]
        
        # uniformity from https://arxiv.org/pdf/2005.10242.pdf
        sq_pdist = torch.pdist(z, p=2).pow(2)
        g_score = sq_pdist.mul(-self.tau).exp().mean()

        return {
            'loss': g_score.log() + 4 * self.tau,
            'contrast_acc': 0.,
            'g_score': g_score.item(),
        }


def exponential_loss_transform(l, shift=0., scale=1.):
    '''
    Apply exponential transform to the loss l to make it assymetric and weight lower values more
    '''
    return -torch.exp((-scale * (l - shift)).clamp_max(5))


def softplus_loss_transform(l, shift=0., scale=1.):
    '''
    Apply exponential transform to the loss l to make it assymetric and weight lower values more
    '''
    return -F.softplus(-scale * (l - shift))

def identical(l):
    return l


def smooth_max(inputs, dim=0, alpha=1.):
    return torch.logsumexp(inputs * alpha, dim) / alpha


class ASUniformityTraining(pl.LightningModule):
    def __init__(
            self,
            task_temp=0.5,
            as_loss='as',
            real_incl=False,
            as_loss_coef=1.,
            coverage_eval=True,
            arch='resnet18',
            as_arch='resnet18',
            as_eval_max_epochs=100,
            as_test_max_epochs=100,
            as_loss_exp_transform='',
            proj='linear',
            encoder_backbone_ckpt='',
            freeze_backbone=False,
            diversity_loss='uniformity',
            train_linear_tasks=False,
            task_on_factors=False,
            n_classes=2,
            **kwargs
        ):
        super().__init__()
        self.save_hyperparameters() # a method from pl.LightningModule. Save the hyperparams.
        print(f'{self.hparams=}')

        kwargs['return_indicies'] = True
        # DataModule according to the self.hparams.dataset which is specified when we construct this class object
        if self.hparams.dataset == 'cifar10':
            tmp_dm = MyCIFAR10DataModule
        elif self.hparams.dataset == 'tiny_imagenet':
            tmp_dm = TinyImageNetDataModule
        elif self.hparams.dataset == 'tiny_imagenet_64':
            tmp_dm = partialclass(TinyImageNetDataModule, image_size=64)
        # elif 'esc50' in self.hparams.dataset:
        #     tmp_dm = ESC50DataModule
        elif 'audioset' in self.hparams.dataset:
            tmp_dm = AudiosetDataModule
        else:
            raise NotImplementedError

        # Set self.data_module
        # if self.hparams.dataset == 'esc50':
        #     self.data_module = tmp_dm(
        #         dataset_path='/mnt/esc50',  # os.environ.get('/mnt/esc50', os.getcwd()),
        #         image_size=64,    
        #         **{k: v for k, v in kwargs.items() if
        #            k not in ['data_dir', 'dataset_path', 'val_split', 'n_classes']}
        #     )
        # elif self.hparams.dataset == 'esc50human':
        #     self.data_module = tmp_dm(
        #         dataset_path='/mnt/esc50',  # os.environ.get('/mnt/esc50', os.getcwd()),
        #         image_size=64,
        #         selective='esc50human',
        #         **{k: v for k, v in kwargs.items() if
        #            k not in ['data_dir', 'dataset_path', 'val_split', 'n_classes', 'selective']}
        #     )
        if self.hparams.dataset == 'audioset':
            self.data_module = tmp_dm(
                # TODO Set to the appropriate path to audioset dataset
                dataset_path='YOUR_AUDIO_DIR',
                image_size=64,
                **{k: v for k, v in kwargs.items() if
                   k not in ['data_dir', 'dataset_path', 'val_split', 'n_classes']}
            )
        elif self.hparams.dataset in ['audiosetpop', 'audiosetclassical', 'audiosetoutside_urban_or_manmade',
                                      'audiosetoutside_rural_or_natural', 'audiosetdomestic_sounds_home_sounds',
                                      'audiosetspeech',
                                      'audiosetprocessed_speech', 'audiosetspeech_single_label']: # TODO: Edit as needed
            selective_mapping = {
                'audiosetpop': 'pop_music',
                'audiosetclassical': 'classical_music',
                'audiosetoutside_urban_or_manmade': 'outside_urban_or_manmade',
                'audiosetoutside_rural_or_natural': 'outside_rural_or_natural',
                'audiosetdomestic_sounds_home_sounds': 'domestic_sounds_home_sounds',
                'audiosetspeech': 'speech',
                'audiosetspeech_single_label': 'speech_single_label',
                'audiosetprocessed_speech': 'processed_speech'
            }

            selected_selective = selective_mapping[self.hparams.dataset]
            self.data_module = tmp_dm(
                # TODO Set to the appropriate path to audioset dataset
                dataset_path='YOUR_AUDIO_DIR',
                image_size=64,
                selective=selected_selective,
                **{k: v for k, v in kwargs.items() if
                   k not in ['data_dir', 'dataset_path', 'val_split', 'n_classes', 'selective']}
            )
        else:
            self.data_module = tmp_dm(
                data_dir=os.environ.get('DATA_ROOT', os.getcwd()),
                **{k: v for k, v in kwargs.items() if k not in ['data_dir', 'n_classes']},
            )
    
        self.data_module.setup()
        self._trainloader = self.data_module.train_dataloader()
        self._valloader = self.data_module.val_dataloader()

        # Uniformity: a contrastive loss based on the uniformity of embedding vectors
        self.uniformity_loss = Uniformity(
            self.hparams.temperature,   # default: 2
            distributed=False,
        )

        # TaskdiscoveryTODO: change the args with 'as_' prefix here
        self.hparams.in_dim = self.data_module.dims[0]
        if self.hparams.as_loss == 'as':
            self.agreement_loss = AgreementLoss(**self.hparams)
        elif self.hparams.as_loss == 'train-loss-proxy':
            self.agreement_loss = TrainingProxyASLoss(**self.hparams)
        elif self.hparams.as_loss == 'xas':
            assert self.hparams.data_mode == 'full'
            self.agreement_loss = TrainingXASLoss(**self.hparams)
        elif self.hparams.as_loss == 'foxas':
            assert self.hparams.data_mode == 'full'
            assert not self.hparams.real_incl
            self.agreement_loss = FOXASLoss(**self.hparams)
            self.automatic_optimization = False
        else:
            raise ValueError(f'{self.hparams.as_loss=}')


        self.as_loss_transform = identical
        if self.hparams.as_loss_exp_transform == 'exp':
            self.as_loss_transform = partial(
                exponential_loss_transform,
                shift=self.hparams.as_exp_transform_shift,
                scale=self.hparams.as_exp_transform_scale,
            )
        elif self.hparams.as_loss_exp_transform == 'softplus':
            self.as_loss_transform = partial(
                softplus_loss_transform,
                shift=self.hparams.as_exp_transform_shift,
                scale=self.hparams.as_exp_transform_scale,
            )
        elif self.hparams.as_loss_exp_transform != '':
            raise ValueError(f'{self.hparams.as_loss_exp_transform=}')

        print(f'{self.data_module.dims=}')
        in_dim = self.data_module.dims
        if self.hparams.task_on_factors:
            in_dim = self.data_module.factors_dim

        # class TaskDiscoveryEncoder(nn.Module)
        self.encoder = TaskDiscoveryEncoder(
            in_dim=in_dim,
            h_loss=lambda h: self.uniformity_loss(h)['loss'],
            h_dim=self.hparams.h_dim,
            arch=self.hparams.arch,
            proj=self.hparams.proj,
            freeze_backbone=freeze_backbone,
            n_classes=self.hparams.n_classes,
        )
        print(f'===> Task-Encoder:\n{self.encoder}')

        # Load checkpoint
        if self.hparams.encoder_backbone_ckpt != "":
            print(f"===> self.hparams.encoder_backbone_ckpt:{self.hparams.encoder_backbone_ckpt}")
            assert not self.hparams.normalize, 'The backbone is usually SSL, which is pre-trained w/o normalziation'
            backbone_ckpt = torch.load(self.hparams.encoder_backbone_ckpt, map_location=self.device)
            msg = self.encoder.backbone.load_state_dict(backbone_ckpt, strict=False)
            assert len(msg.missing_keys) == 2

        if self.hparams.freeze_backbone:    # default value is false
            print(f"===> self.hparams.freeze_backbone: {self.hparams.freeze_backbone}")
            for p in self.encoder.backbone.parameters():
                p.requires_grad = False
            self.encoder.backbone.eval()


        if self.hparams.n_linear_tasks != -1:
            print(f"===> self.hparams.n_linear_tasks: {self.hparams.n_linear_tasks}")
            if self.hparams.diversity_loss == 'uniformity':
                print(f"===> self.hparams.diversity_loss: {self.hparams.diversity_loss}")
                assert self.encoder.h_dim >= self.hparams.n_linear_tasks
                self.linear_tasks = nn.parameter.Parameter(
                    torch.FloatTensor(utils.rvs(self.encoder.h_dim)[:self.hparams.n_linear_tasks]),
                    requires_grad=self.hparams.train_linear_tasks,
                ) # rvs(): Generate a set of mutually perpendicular vectors in a given Euclidean space dimension
            elif self.hparams.diversity_loss == 'mi':
                print(f"===> self.hparams.diversity_loss: {self.hparams.diversity_loss}")
                self.linear_tasks = nn.parameter.Parameter(
                    torch.randn(self.hparams.n_linear_tasks, self.encoder.h_dim),
                    requires_grad=self.hparams.train_linear_tasks,
                )
                self.linear_tasks_bias = nn.parameter.Parameter(
                    torch.zeros((self.hparams.n_linear_tasks,)),
                    requires_grad=self.hparams.train_linear_tasks,
                )
            else:
                raise ValueError(f'{self.hparams.diversity_loss=}')

            # a sequence of numbers that represent different linear tasks.
            self.tasks_dataset = TensorDataset(torch.arange(self.hparams.n_linear_tasks))
            # self._as_scores will be updated in further steps
            self._as_scores = torch.zeros(self.hparams.n_linear_tasks) - 1
        else:
            self.tasks_dataset = TensorDataset(torch.zeros(1).long())
            self._as_scores = torch.zeros(1) - 1

        if self.hparams.n_classes == 2:
            print(f"===> self.hparams.dataset: {self.hparams.dataset}")
            # if self.hparams.dataset == 'esc50':
            #     used_classes = set(self.hparams.get('include_classes', None) or set((range(50))))
            #     print(f'====> used_classes == set((range(50)) -> {used_classes == set((range(50)))}')
            #     # remove trivial tasks where all used esc50 classes are in a single class
            #     valid_real_tasks = utils.get_main_tasks_idxs_from_included_classes(used_classes)
            #     print(f'===> {len(valid_real_tasks)} {valid_real_tasks=}')
            #     self._real_tasks = np.array(
            #         [ESC50ClassificationTask(task_type='real', task_idx=i, dataset=self.hparams.dataset) for i in valid_real_tasks]
            #     )
            # elif self.hparams.dataset == 'esc50human':
            #     used_classes = set(self.hparams.get('include_classes', None) or set(ESC50HUMAN_CLASSES)) # real class idx (a.k.a. label)
            #     # remove trivial tasks where all used esc50 classes are in a single class
            #     print(f'====> {used_classes=}')
            #     valid_real_tasks = utils.get_main_tasks_idxs_from_included_classes(used_classes, self.hparams.dataset)
            #     print(f'===> {len(valid_real_tasks)} {valid_real_tasks=}')
            #     self._real_tasks = np.array(
            #         [ESC50ClassificationTask(task_type='real', task_idx=i, dataset=self.hparams.dataset) for i in valid_real_tasks]
            #     )   # labels are replaced with tentative class indices (e.g., 0, 1, 2,...)
            # TODO if self.hparams.dataset == 'audioset':
            if self.hparams.dataset in ['audiosetpop', 'audiosetclassical']:
                used_classes = set(self.hparams.get('include_classes', None) or (0, 1)) # FIXME: refer to the code below
                valid_real_tasks = utils.get_main_tasks_idxs_from_included_classes(used_classes, self.hparams.dataset)
                print(f'===> {len(valid_real_tasks)} {valid_real_tasks=}')
                self._real_tasks = np.array(
                    [AudiosetClassificationTask(task_type='real', task_idx=i, dataset=self.hparams.dataset) for i in
                     valid_real_tasks]
                )
            elif self.hparams.dataset in ['audiosetoutside_urban_or_manmade', 'audiosetoutside_rural_or_natural',
                                          'audiosetdomestic_sounds_home_sounds', 'audiosetspeech', 'audiosetspeech_single_label']:
                include_classes_dict = {'audiosetdomestic_sounds_home_sounds': set(range(25)),  # the constants are the unique classes in the dataset
                                        'audiosetoutside_urban_or_manmade': set(range(25)),    # 127
                                        'audiosetoutside_rural_or_natural' : set(range(25)),   # 154
                                        'audiosetspeech': set(range(4)),
                                        'audiosetspeech_single_label': set(range(4))}
                #used_classes = set(self.hparams.get('include_classes', None) or (0, 1))
                used_classes = include_classes_dict
                valid_real_tasks = utils.get_main_tasks_idxs_from_included_classes(used_classes[self.hparams.dataset], self.hparams.dataset)
                print(f'===> {len(valid_real_tasks)} {valid_real_tasks=}')
                self._real_tasks = np.array(
                    [AudiosetClassificationTask(task_type='real', task_idx=i, dataset=self.hparams.dataset) for i in
                     valid_real_tasks]
                )
            else:   #self.hparams.dataset == 'cifar10':
                used_classes = set(self.hparams.get('include_classes', None) or (1, 2, 3, 4, 5, 6, 7, 8, 9))
                # remove trivial tasks where all used cifar classes are in a single class
                valid_real_tasks = utils.get_main_tasks_idxs_from_included_classes(used_classes)
                print(f'===> {len(valid_real_tasks)} {valid_real_tasks=}')
                self._real_tasks = np.array(
                    [CIFARClassificationTask(task_type='real', task_idx=i, dataset=self.hparams.dataset) for i in valid_real_tasks]
                )
        else:
            assert not self.hparams.coverage_eval

        if self.hparams.real_incl:
            print(f"===> self.hparams.real_incl: {self.hparams.real_incl}")
            if self.hparams.real_incl_group == '':
                if self.hparams.n_real_incl < 126:  # the desired number of tasks to be included
                    self._real_tasks_included, self._real_tasks_not_included = train_test_split(
                        np.arange(len(self._real_tasks)),
                        train_size=self.hparams.n_real_incl,
                        random_state=self.hparams.real_incl_split_seed,
                    )
                else:
                    self._real_tasks_included, self._real_tasks_not_included = np.arange(len(self._real_tasks)), []
            else:
                # FixMe there is no assets/tasks/cifar-groupped-splits.csv
                split = pd.read_csv('assets/tasks/cifar-groupped-splits.csv', index_col=0)
                self._real_tasks_included = split[split[self.hparams.real_incl_group]].index.values
                self._real_tasks_not_included = split[~split[self.hparams.real_incl_group]].index.values

            self.real_clf = nn.Linear(self.encoder.h_dim, len(self._real_tasks_included))
            self._real_tasks_loader = self.data_module.train_dataloader(batch_size=2*self.hparams.batch_size)
            self._real_tasks_iterator = iter(self._real_tasks_loader)

        self._logistic_regr = NeuralNetClassifier(
            nn.Linear(self.encoder.h_dim, len(self._real_tasks) if self.hparams.n_classes == 2 else self.hparams.n_classes),
            max_epochs=20,
            lr=0.001,
            optimizer=torch.optim.Adam,
            criterion=nn.BCEWithLogitsLoss if self.hparams.n_classes == 2 else nn.CrossEntropyLoss,
            iterator_train__shuffle=True,
            #device='cuda',
            train_split=None,
        )

        #print(f"===> After self._logistic_regr: {self._logistic_regr}")
        self._as_df = pd.DataFrame()

        self._as_eval_stopping_threshold = 0.98
        self._as_eval_max_epochs = self.hparams.as_eval_max_epochs

        if self.hparams.diversity_loss == 'mi':
            setattr(self._task_forward_with_mutial_info.__func__, "eval", self.encoder.eval)
            setattr(self._task_forward_with_mutial_info.__func__, "train", self.encoder.train)
        self._last_probs_hist = None

        self._task_idx = 0

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = AgreementLoss.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--temperature', type=float, default=2.)
        parser.add_argument('--uniformity_coef', type=float, default=1.)
        parser.add_argument('--uniformity_pretraining_steps', type=int, default=0)
        parser.add_argument('--diversity_loss', type=str, default='uniformity')
        parser.add_argument('--mi_reg_coef', type=float, default=0.)
        parser.add_argument('--mi_ce_coef', type=float, default=0.)

        parser.add_argument('--n_linear_tasks', type=int, default=-1)
        parser.add_argument('--dataset', type=str, default='cifar10')
        parser.add_argument('--h_dim', type=int, default=512)
        parser.add_argument('--encoder_backbone_ckpt', type=str, default='')
        parser.add_argument('--proj', type=str, default='linear')
        parser.add_argument('--freeze_backbone', action='store_true', default=False)
        parser.add_argument('--train_linear_tasks', action='store_true', default=False)
        parser.add_argument('--task_on_factors', action='store_true', default=False)
        parser.add_argument('--n_classes', type=int, default=2)

        parser.add_argument('--as_arch', type=str, default='resnet18')
        parser.add_argument('--task_temp', type=float, default=0.5)
        parser.add_argument('--as_loss_coef', type=float, default=1.)
        parser.add_argument('--as_loss', type=str, default='as')
        parser.add_argument('--as_loss_exp_transform', type=str, default='')
        parser.add_argument('--as_exp_transform_scale', type=float, default=20)
        parser.add_argument('--as_exp_transform_shift', type=float, default=0.6)

        parser.add_argument('--as_eval_max_epochs', type=int, default=30)
        parser.add_argument('--as_eval_n_tasks', type=int, default=5)
        parser.add_argument('--as_test_max_epochs', type=int, default=100)
        parser.add_argument('--as_test_n_tasks', type=int, default=-1)

        parser.add_argument('--real_incl', action='store_true', default=False)
        parser.add_argument('--real_incl_split_seed', type=int, default=0)
        parser.add_argument('--n_real_incl', type=int, default=126)
        parser.add_argument('--real_incl_group', type=str, default='')
        parser.add_argument('--no_real_incl', dest='real_incl', action='store_false', default=False)
        parser.add_argument('--real_incl_locc_coef', type=float, default=1.)

        parser.add_argument('--early_stop_xas_threshold', type=float, default=-1)
        parser.add_argument('--early_stop_xas_n_steps', type=int, default=-1)
        parser.add_argument('--min_nsteps', type=int, default=0)

        parser.add_argument('--coverage_eval', dest='coverage_eval', action='store_true', default=True)
        parser.add_argument('--no_coverage_eval', dest='coverage_eval', action='store_false', default=True)
        return parser

    def _mutial_information_dissimilarity_loss(self, logits):
        # logits BxH
        probs = F.softmax(logits, dim=2)
        self._last_probs_hist = wandb.Histogram(utils.tonp(probs).flatten())

        return self._mutual_info_loss(probs)

    def _task_forward_with_mutial_info(self, x, y=None, idxs=None, factors=None, get_loss=None):
        if factors is not None:
            h = self.encoder(factors, out='h', get_loss=False)[0]
        else:
            h = self.encoder(x, out='h', get_loss=False)[0]

        logits = self.hparams.task_temp * (h @ self.linear_tasks.t() + self.linear_tasks_bias[None])
        logits = torch.stack([logits, -logits]).transpose(0, 1).transpose(1, 2) # -> BxHx2
        div_loss = self._mutial_information_dissimilarity_loss(logits)
        div_loss += self.hparams.mi_ce_coef * entropy_with_logits(logits).mean()
        return logits[:, self._task_idx], div_loss

    def _mutual_info_loss(self, probs):
        # from https://arxiv.org/pdf/2202.03418.pdf
        """ Input: predicted probabilites on target batch. """
        B, H, D = probs.shape # B=batch_size, H=heads, D=pred_dim
        marginal_p = probs.mean(dim=0).clamp_min(1e-9) # H, D
        reg = (marginal_p * (marginal_p.log() - np.log(1./marginal_p.shape[1]))).sum(1).mean()
        marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p) # H, H, D, D
        marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)") # H^2, D^2
        joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(dim=0).clamp_min(1e-9) # H, H, D, D
        joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)") # H^2, D^2
        kl_divs = joint_p * (joint_p.log() - marginal_p.log())
        kl_grid = rearrange(kl_divs.sum(dim=-1), "(h g) -> h g", h=H) # H, H
        idxs = torch.triu_indices(H, H, offset=1)
        kl_loss = torch.log(kl_grid[idxs[0], idxs[1]].clamp_min(1e-9))
        assert kl_loss.dim() == 1
        return smooth_max(kl_loss) + reg * self.hparams.mi_reg_coef

    def _mayby_log_to_experiment(self, logs):
        if self.logger is not None:
            self.logger.experiment.log(logs)

    def on_train_start(self) -> None:
        if self.hparams.real_incl:
            self._mayby_log_to_experiment({
                'real_incl/included_tasks': wandb.Table(dataframe=DataFrame({'task_idx': self._real_tasks_included})),
                'real_incl/not_included_tasks': wandb.Table(dataframe=DataFrame({'task_idx': self._real_tasks_not_included})),
            })

    def set_task(self, idx=None, w=None):
        assert idx is None or w is None

        # print(f'{self.hparams.n_linear_tasks=}')
        if self.hparams.n_linear_tasks == -1:
            self.set_random_task(seed=idx)
            return

        self._task_idx = idx
        if w is None:
            w = self.linear_tasks[idx]
        # w is a single hyperplane, but we need two for softmax
        self.encoder.classifier.weight.copy_(self.hparams.task_temp * torch.stack([w, -w]))

    def set_random_task(self, seed=None):
        ws = utils.random_k_way_linear_task(self.hparams.n_classes, self.hparams.h_dim, seed)
        self.encoder.classifier.weight.copy_(self.hparams.task_temp * torch.FloatTensor(ws.T))

    def logits_all_tasks(self, input):
        if self.hparams.diversity_loss == 'uniformity':
            h = self.encoder(input, out='h', get_loss=False)[0]
            return h @ self.linear_tasks.t()
        elif self.hparams.diversity_loss == 'mi':
            h = self.encoder(input, out='h', get_loss=False)[0]
            logits = self.hparams.task_temp * (h @ self.linear_tasks.t() + self.linear_tasks_bias[None])
            return logits

    def predict_all_tasks(self, input):
        logits = self.logits_all_tasks()
        return (logits > 0).long()

    def _get_next_training_batch(self):
        try:
            batch = next(self._trainiterator)
        except StopIteration:
            self._trainiterator = iter(self._trainloader)
            batch = next(self._trainiterator)

        return batch

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        if self._last_probs_hist is not None:
            self._mayby_log_to_experiment({'task_prob_hist': self._last_probs_hist})

    def training_step(self, batch, batch_idx):
        logs = {
            'task_idx': -1,
        }
        task_idx = -1

        if self.hparams.as_loss_coef != 0:
            # set linear task layer
            if self.hparams.n_linear_tasks != -1:
                # set one of the predifined tasks
                task_idx = batch[0][0]
                self.set_task(task_idx)
                logs['task_idx'] = task_idx
            else:
                # sample random linear task
                self.set_random_task()

            if self.hparams.as_loss == 'foxas':
                return self._foxas_train_step(logs)

            # get agreement loss and metrics
            logs = self.agreement_loss(
                self.encoder if self.hparams.diversity_loss != 'mi' else self._task_forward_with_mutial_info,
                self._trainloader,
                self._valloader if self.hparams.as_loss_coef != 0 else None,
            )
        else:
            logs['loss'] = torch.zeros(1).to(self.device)

            batch = self._get_next_training_batch()
            x = batch[0].to(self.device)
            logs['h_loss'] = self._task_forward_with_mutial_info(x)[1]

        logs['ag_loss'] = logs['loss'].item()
        logs['loss'] = self.as_loss_transform(logs['loss']) * self.hparams.as_loss_coef + self.hparams.uniformity_coef * logs['h_loss']
        logs['uniformity_loss'] = logs.pop('h_loss').item()

        self._as_scores[task_idx] = logs.get('agreement_acc', 0)

        if self.hparams.real_incl:
            real_incl_logs = self._real_incl_loss()
            logs['loss'] = logs['loss'] + self.hparams.real_incl_locc_coef * real_incl_logs.pop('loss')
            if self.hparams.as_loss_coef == 0 and self.hparams.uniformity_coef != 0:
                logs['loss'] = logs['loss'] + self.hparams.uniformity_coef * real_incl_logs['h_loss']
                logs['uniformity_loss'] = real_incl_logs.pop('h_loss').item()
            logs.update(real_incl_logs)

        self.log_dict(logs)

        return logs

    def _foxas_train_step(self, logs):
        # reset iterator manualy to avoid repeted batches due to a "cycle"
        self._trainiterator = iter(self._trainloader)
        prev_batch = self._get_next_training_batch()
        xas_acc = []

        for i in range(self.hparams.nsteps):
            opt = self.optimizers()
            opt.zero_grad()
            new_batch = self._get_next_training_batch()
            inner_logs = self.agreement_loss(
                self.encoder,
                prev_batch,
                new_batch,
                self._foxasloss_backward,
                reset=(i==0),
            )
            prev_batch = new_batch
            self._mayby_log_to_experiment(inner_logs)

            # early stopping logic
            if self.hparams.early_stop_xas_n_steps != -1:
                assert self.hparams.early_stop_xas_threshold != -1
                xas_acc.append(inner_logs['models_val/acc'])
                xas_acc = xas_acc[-self.hparams.early_stop_xas_n_steps:]
                if len(xas_acc) == self.hparams.early_stop_xas_n_steps and (np.array(xas_acc) > self.hparams.early_stop_xas_threshold).all() and i >= self.hparams.min_nsteps:
                    break

        logs.update({f'last_inner_step/{k}': v for k, v in inner_logs.items()})
        logs['uniformity_loss'] = logs['last_inner_step/h_loss'].item()
        logs['agreement_acc'] = logs['last_inner_step/models_val/acc']
        logs['agreement_loss'] = logs['last_inner_step/models_val/loss'].item()
        logs['steps'] = i
        self._as_scores[logs['task_idx']] = logs.get('agreement_acc', 0)

        self.log_dict(logs)
        return logs

    def _foxasloss_backward(self, as_loss, h_loss):
        # assume zero grad is done
        loss = self.as_loss_transform(as_loss) * self.hparams.as_loss_coef + self.hparams.uniformity_coef * h_loss
        self.manual_backward(loss)
        opt = self.optimizers()
        opt.step()

    def _real_incl_loss(self):
        try:
            x, y, _ = next(self._real_tasks_iterator)
        except StopIteration:
            self._real_tasks_iterator = iter(self._real_tasks_loader)
            x, y, _ = next(self._real_tasks_iterator)

        x = x.to(self.device)
        # get labels for all the real tasks based on the original ys
        incl_tasks = self._real_tasks[self._real_tasks_included]
        y = torch.stack([t(y=y) for t in incl_tasks]).t().float().to(self.device)

        get_loss = (self.hparams.as_loss_coef == 0 and self.hparams.uniformity_coef != 0)
        h, h_loss = self.encoder(x, get_loss=get_loss, out='h')
        p = self.real_clf(h)

        return {**self._real_incl_metrics(p, y), 'h_loss': h_loss}
    
    @staticmethod
    def _real_incl_metrics(p, y, prefix=''):
        logs = {}
        logs[f'loss'] = F.binary_cross_entropy_with_logits(p, y)
        logs[f'real_incl/{prefix}loss'] = logs['loss'].item()

        acc = ((p > 0).long() == y).float().mean(0)
        logs[f'real_incl/{prefix}acc_median'] = torch.median(acc).item()
        logs[f'real_incl/{prefix}acc_q1'] = torch.quantile(acc, 0.25).item()
        logs[f'real_incl/{prefix}acc_q3'] = torch.quantile(acc, 0.75).item()
        logs[f'real_incl/{prefix}acc_min'] = torch.min(acc).item()
        logs[f'real_incl/{prefix}acc_max'] = torch.max(acc).item()

        return logs

    def configure_optimizers(self):
        train_parameters = tuple(p for p in self.encoder.parameters() if p.requires_grad)
        if self.hparams.real_incl:
            train_parameters = train_parameters + tuple(p for p in self.real_clf.parameters())
        if self.hparams.train_linear_tasks:
            train_parameters = train_parameters + (self.linear_tasks, self.linear_tasks_bias)
        opt = torch.optim.Adam(train_parameters, lr=self.hparams.encoder_learning_rate)
        print(f'========> Task Optimizer: \n {opt}')
        return opt

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        self._mayby_log_to_experiment({'as_histogram': wandb.Histogram(self._as_scores)})

    def _get_outputs(self, loader):
        out = defaultdict(list)
        for batch in loader:
            batch = [b.to(self.device) for b in batch]
            x = batch[0] if not self.hparams.task_on_factors else batch[3]
            with torch.no_grad():
                out['h'].append(self.encoder(x, out='h', get_loss=False)[0])
                out['y'].append(torch.stack([t(y=batch[1]) for t in self._real_tasks]).t().float().to(self.device))

                if self.hparams.real_incl:
                    out['p'].append(self.real_clf(out['h']))
        return {k: torch.cat(v) for k, v in out.items()}

    def validation_step(self, batch, batch_idx):
        # self.encoder.train()
        x = batch[0] if not self.hparams.task_on_factors else batch[3]

        out = {}
        with torch.no_grad():
            out['h'] = self.encoder(x, out='h', get_loss=False)[0]
            out['y'] = torch.stack([t(y=batch[1]) for t in self._real_tasks]).t().float().to(self.device)

            if self.hparams.real_incl:
                out['p'] = self.real_clf(out['h'])

        return out

    def validation_epoch_end(self, outputs) -> None:
        # TaskdiscoveryTODO: Unify with testing
        out = {}
        for k in outputs[0].keys():
            out[k] = torch.cat([a[k] for a in outputs], dim=0)

        logs = self._eval_tasks_similarity(out['h'])

        if self.hparams.coverage_eval:
            val_out = self._get_outputs(self.data_module.val_dataloader(
                batch_size=min(len(self.data_module.dataset_val), 5*self.hparams.batch_size),
            ))
            logs.update(self._eval_coverage(out['h'], out['y'], val_embs=val_out['h'], val_y=val_out['y']))

        # evaluate and log AS in "standard" setting
        logs.update(self._eval_as_on_tasks())

        if self.hparams.real_incl:
            real_incl_logs = self._real_incl_metrics(out['p'], out['y'][:, self._real_tasks_included], prefix='val_')
            real_incl_logs.pop('loss')
            self.log_dict(real_incl_logs)

        self._mayby_log_to_experiment(logs)

    def _eval_tasks_similarity(self, embs):
        logs = {}

        print(f'===> {self.hparams.n_linear_tasks=}')
        if self.hparams.n_linear_tasks == -1:
            linear_tasks = np.stack([
                utils.random_k_way_linear_task(self.hparams.n_classes, self.hparams.h_dim, i) for i in range(self.hparams.h_dim)
            ]).transpose((1, 0, 2))
            tasks = torch.tensordot(embs, torch.FloatTensor(linear_tasks).to(self.device), dims=1).argmax(2).long().cpu()
            s = utils.hamming_sym(tasks.t(), binary=False)
        else:
            tasks = (embs @ self.linear_tasks.t() > 0).long().cpu()
            s = utils.hamming_sym(tasks.t())

        np.fill_diagonal(s, np.nan)

        mi_s = pairwise_distances(tasks.t(), metric=adjusted_mutual_info_score, n_jobs=-1)
        np.fill_diagonal(mi_s, np.nan)

        logs.update({
            'tasks_similarity/heatmap': self._get_heatmap_fig(s),
            'tasks_similarity/hist': wandb.Histogram(s[np.tril_indices_from(s, k=-1)]),
            'tasks_similarity/table': wandb.Table(dataframe=DataFrame(s)),

            'tasks_similarity/mi_heatmap': self._get_heatmap_fig(mi_s, 0, 1),
            'tasks_similarity/mi_hist': wandb.Histogram(mi_s[np.tril_indices_from(mi_s, k=-1)]),
            'tasks_similarity/mi_table': wandb.Table(dataframe=DataFrame(mi_s)),
        })

        return logs

    def _eval_as_on_tasks(self, task_idxs=None):
        if self.hparams.as_eval_n_tasks == 0: return {}

        if task_idxs is None:
            if self.hparams.n_linear_tasks != -1:
                all_task_idxs = np.arange(32)   # Generate an array containing the indices from 0 to 31
                np.random.shuffle(all_task_idxs) # Randomly shuffle the array # Not sure whether this is mandatory or not
                task_idxs = all_task_idxs[:]

                # # sample random task indicies to eval
                # task_idxs = np.random.choice(
                #     np.arange(self.hparams.n_linear_tasks),
                #     size=(min(self.hparams.as_eval_n_tasks, self.hparams.n_linear_tasks),),
                #     replace=False,
                # )
            else:
                task_idxs = [None] * self.hparams.as_eval_n_tasks

        df = []
        for task_idx in task_idxs:
            self.set_task(idx=task_idx)
            logs = self._eval_as()
            logs['task_idx'] = task_idx
            df.append(logs)
            # self.log_dict({f'eval_as/{k}': v for k, v in logs.items()})
        df = DataFrame(df)
        df['step'] = self.global_step
        df['epoch'] = self.current_epoch

        # log the whole AS table
        self._as_df = pd.concat([self._as_df, df], axis=0)
        if self.logger is not None:
            self._as_df.to_csv(os.path.join(self.logger.experiment.dir, 'as_eval_df.csv'))
            wandb.save(os.path.join(self.logger.experiment.dir, 'as_eval_df.csv'))

        return {
            'eval_as/table': wandb.Table(dataframe=df),
            'eval_as/val_acc_agreement': wandb.Histogram(df['val_acc_agreement']),
        }

    def on_test_start(self) -> None:
        # change params to train networks for AS for longer
        self._as_eval_stopping_threshold = 2
        self._as_eval_max_epochs = self.hparams.as_test_max_epochs

    def on_test_epoch_start(self) -> None:
        train_out = self._get_outputs(self.data_module.train_dataloader(
            batch_size=min(len(self.data_module.dataset_train), 5*self.hparams.batch_size),
        ))
        val_out = self._get_outputs(self.data_module.val_dataloader(
            batch_size=min(len(self.data_module.dataset_val), 5*self.hparams.batch_size),
        ))

        logs = {}
        logs.update(self._eval_tasks_similarity(train_out['h']))
        if self.hparams.coverage_eval:
            logs.update(self._eval_coverage(train_out['h'], train_out['y'], val_embs=val_out['h'], val_y=val_out['y']))

        if self.hparams.real_incl:
            real_incl_logs = self._real_incl_metrics(val_out['p'], val_out['y'][:, self._real_tasks_included], prefix='val_')
            real_incl_logs.pop('loss')
            self.log_dict(real_incl_logs)
        
        self._mayby_log_to_experiment(logs)

    def test_step(self, batch, batch_idx) -> None:
        task_idx = batch[0][0].long().item()
        self.set_task(task_idx)
        logs = self._eval_as()
        logs['task_idx'] = task_idx
        self._mayby_log_to_experiment({f'test_as/{k}': v for k, v in logs.items()})
        return logs

    def test_epoch_end(self, outputs) -> None:
        print(outputs)
        out = {}
        for k in outputs[0].keys():
            out[k] = np.concatenate([utils.tonp(a[k])[None] for a in outputs])

        df = pd.DataFrame(out)
        self._mayby_log_to_experiment({
            'test_as/table': wandb.Table(dataframe=df),
            'test_as/hist': wandb.Histogram(df['val_acc_agreement']),
        })

    def _mi_task_forward_for_as_eval(self, *args, **kwargs):
        return self._task_forward_with_mutial_info(*args, **kwargs)[0].argmax(1)

    def _eval_as(self):
        in_dim = self.data_module.dims if not self.hparams.task_on_factors else self.data_module.factors_dim
        # TODO: Check if this work well
        if 'audioset' in self.hparams.dataset:
            # ['audiosetoutside_urban_or_manmade', 'audiosetoutside_rural_or_natural', 
            # 'audiosetdomestic_sounds_home_sounds', 'audiosetspeech', 'audiosetpop', 'audiosetclassical']
            print("as_uniformity.py ::: _eval_as() AudiosetEmbeddingClassificationTask")
            task = AudiosetEmbeddingClassificationTask(
                h_dim=self.encoder.h_dim,
                in_dim=in_dim,
                out_type='class',
                arch=self.hparams.arch,
                proj=self.hparams.proj,
                n_classes=self.hparams.n_classes,
            )
        # elif 'esc50' in self.hparams.dataset:
        #     print("as_uniformity.py :::  _eval_as() ESC50EmbeddingClassificationTask")
        #     task = ESC50EmbeddingClassificationTask(
        #         h_dim=self.encoder.h_dim,
        #         in_dim=in_dim,
        #         out_type='class',
        #         arch=self.hparams.arch,
        #         proj=self.hparams.proj,
        #         n_classes=self.hparams.n_classes,
        #     )
        else:
            print("as_uniformity.py :::  _eval_as() CIFAREmbeddingClassificationTask")
            task = CIFAREmbeddingClassificationTask(
                h_dim=self.encoder.h_dim,
                in_dim=in_dim,
                out_type='class',
                arch=self.hparams.arch,
                proj=self.hparams.proj,
                n_classes=self.hparams.n_classes,
            )
    
        if self.hparams.diversity_loss == 'mi':
            task.forward = self._mi_task_forward_for_as_eval

        task.eval()
        for p in task.parameters():
            p.requires_grad = False
        task.encoder.load_state_dict(self.encoder.state_dict())

        agreement_score = ClassificationAgreementScore()
        as_eval_model = TwoSupervisedModels(
            in_dim=self.hparams.in_dim,
            data_mode='full',
            agreement_score=agreement_score,
            task=task,
            arch=self.hparams.arch,
        )

        # stop training when if task is fitted earlier
        stop_callback = pl.callbacks.EarlyStopping(
            monitor='acc_0',
            mode='max',
            stopping_threshold=self._as_eval_stopping_threshold,
            check_on_train_epoch_end=True,
            patience=int(1e6),
        )
        trainer = pl.Trainer(
            gpus= [4], #torch.cuda.device_count(), #1
            logger=None,
            log_every_n_steps=-1,
            max_epochs=self._as_eval_max_epochs,
            checkpoint_callback=False,
            callbacks=[stop_callback],
        )

        trainer.fit(
            as_eval_model,
            train_dataloader=self.data_module.train_dataloader(persistent_workers=True)
        )
        train_res = trainer.logged_metrics

        val_res = trainer.test(as_eval_model, test_dataloaders=self.data_module.val_dataloader(
            batch_size=min(len(self.data_module.dataset_val), 2*self.hparams.batch_size)
        ))
        test_res = trainer.test(as_eval_model, test_dataloaders=self.data_module.test_dataloader(
            batch_size=min(len(self.data_module.dataset_test), 2*self.hparams.batch_size)
        ))

        del task, trainer, as_eval_model, agreement_score

        return {
            'val_acc_agreement': val_res[0]['test_acc_agreement'],
            'test_acc_agreement': test_res[0]['test_acc_agreement'],
            'train_acc_0': train_res['acc_0'].item(),
            'train_acc_1': train_res['acc_1'].item(),
        }

    def _eval_coverage(self, embs, y, val_embs=None, val_y=None):
        embs = embs.cpu().numpy()
        y = y.cpu().numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(embs).astype(np.float32)

        # TODO: remove ad-hoc
        if self.hparams.n_classes == 10:
            y = y.squeeze().astype(int)

        self._logistic_regr.fit(X, y)
        acc = (self._logistic_regr.predict(X) == y).mean(0)

        if self.hparams.n_classes == 10:
            acc = acc[None]

        logs = {}
        logs['coverage/acc_table'] = wandb.Table(dataframe=DataFrame({'task_idx': np.arange(acc.shape[0]), 'acc': acc}))
        logs['coverage/acc_histogram'] = wandb.Histogram(acc)
        logs['coverage/acc_q1'] = np.quantile(acc, 0.25)
        logs['coverage/acc_q3'] = np.quantile(acc, 0.75)
        logs['coverage/acc_median'] = np.quantile(acc, 0.5)

        if self.hparams.real_incl and len(self._real_tasks_not_included) != 0:
            acc = acc[self._real_tasks_not_included]
            logs['coverage_non_incl/acc_table'] = wandb.Table(dataframe=DataFrame({'task_idx': self._real_tasks_not_included, 'acc': acc}))
            logs['coverage_non_incl/acc_histogram'] = wandb.Histogram(acc)
            logs['coverage_non_incl/acc_q1'] = np.quantile(acc, 0.25)
            logs['coverage_non_incl/acc_q3'] = np.quantile(acc, 0.75)
            logs['coverage_non_incl/acc_median'] = np.quantile(acc, 0.5)
        
        if val_embs is not None:
            val_y = val_y.cpu().numpy()
            val_embs = val_embs.cpu().numpy()
            # TODO: remove ad-hoc
            if self.hparams.n_classes == 10:
                val_y = val_y.squeeze().astype(int)

            acc = (self._logistic_regr.predict(scaler.transform(val_embs)) == val_y).mean(0)
            if self.hparams.n_classes == 10:
                acc = acc[None]

            logs['coverage/val_acc_table'] = wandb.Table(dataframe=DataFrame({'task_idx': np.arange(acc.shape[0]), 'acc': acc}))
            logs['coverage/val_acc_histogram'] = wandb.Histogram(acc)
            logs['coverage/val_acc_q1'] = np.quantile(acc, 0.25)
            logs['coverage/val_acc_q3'] = np.quantile(acc, 0.75)
            logs['coverage/val_acc_median'] = np.quantile(acc, 0.5)

        return logs

    @staticmethod
    def _get_heatmap_fig(s, vmin=0.5, vmax=0.7):
        fig = plt.figure()
        plt.imshow(s, cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.axis('off')
        return fig

    def train_dataloader(self) -> Any:
        return self.task_idxs_dataloader()

    def task_idxs_dataloader(self, shuffle=True, n_tasks=-1) -> Any:
        if n_tasks == -1: n_tasks = len(self.tasks_dataset)

        # TODO: add seed as an hparam
        dataset, _ = torch.utils.data.random_split(
            self.tasks_dataset,
            [n_tasks, len(self.tasks_dataset) - n_tasks],
            generator=torch.Generator().manual_seed(0)
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.data_module.train_dataloader(
            batch_size=min(len(self.data_module.dataset_train), 5*self.hparams.batch_size),
        )

    def test_dataloader(self) -> Any:
        if self.hparams.n_linear_tasks != -1:
            return self.task_idxs_dataloader(shuffle=False, n_tasks=self.hparams.as_test_n_tasks)
        else:
            return DataLoader(
                TensorDataset(torch.zeros(self.hparams.as_test_n_tasks).long()),
                batch_size=1,
                shuffle=False,
                pin_memory=True,
            )


class OptLinearTask(ASUniformityTraining):
    def __init__(self, type='worst', **kwargs):
        kwargs['coverage_eval'] = False
        super().__init__(**kwargs)

        # freeze the encoder part
        for p in self.encoder.parameters():
            p.requires_grad = False

        # unfreeze the linear classifier
        # TODO: maybe add bias term? (it's off in the encoder by default)
        for p in self.encoder.classifier.parameters():
            p.requires_grad = True

    def add_model_specific_args(parser):
        # the args for as-encoder should be loaded from the checkpoint
        parser.add_argument('--type', default='worst')
        parser.add_argument('--as_eval_max_epochs', type=int, default=30)
        parser.add_argument('--as_test_max_epochs', type=int, default=100)
        return parser

    def training_step(self, batch, batch_idx):
        logs = super().training_step(batch, batch_idx)
        # want to maximize the loss to find the worst task
        if self.hparams.type == 'worst':
            logs['loss'] = -logs['loss']
        elif self.hparams.type == 'best':
            pass
        else:
            raise ValueError(f'{self.hparams.type=}')

        return logs

    def configure_optimizers(self):
        # want to train only the linear classifer
        opt = torch.optim.Adam(self.encoder.classifier.parameters(), lr=self.hparams.encoder_learning_rate)
        return opt

    def validation_step(self, *args, **kwargs):
        pass

    def validation_epoch_end(self, outputs) -> None:
        logs = self._eval_as_on_tasks([0])
        self._mayby_log_to_experiment(logs)

    def set_task(self, idx=None, w=None):
        # never change the linear task
        pass

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.task_idxs_dataloader(shuffle=False)

    def task_idxs_dataloader(self, shuffle=True) -> Any:
        return DataLoader(
            self.tasks_dataset[:1],
            batch_size=1,
            shuffle=shuffle,
            pin_memory=True,
        )
