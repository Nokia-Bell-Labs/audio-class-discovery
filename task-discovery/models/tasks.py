import torch
import torch.nn.functional as F
import numpy as np

from abc import abstractmethod
from typing import List, OrderedDict
from torch import nn
from itertools import combinations
from abc import ABC
from torch import Tensor

from .resnet import ResNet8, ResNet18, ResNet50
from .td_encoder import TaskDiscoveryEncoder

import sys, os.path
datamodule_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/datamodule/')
sys.path.append(datamodule_dir)

import random

# TASK BINS
CIFAR_REAL_BIN_TASKS = [cls1 for cls1 in list(combinations(range(10), 5))[:126]]
TINY_IMAGENET_REAL_BIN_TASKS = [] # FixMe = np.load('assets/tasks/tiny_imagenet_binary_tasks.npy').tolist() + [list(range(100)) for _ in range(130)]
CORE_REAL_TASKS_IDX = [0, 11, 18, 27, 31, 38, 40]

def generate_all_binary_tasks(classes):
    for num_classes_in_pair in range(1, len(classes) // 2 + 1):
        for cls1 in combinations(classes, num_classes_in_pair):
            yield cls1


class Task(ABC, nn.Module):
    DIM = None

    def __init__(self):
        super().__init__()

    @abstractmethod    
    def loss(self, a, b):
        pass

    @abstractmethod
    def metrics(self, a, b):
        pass

class BaseClassificationTask(Task):
    def loss(self, prediction, target):
        # print(f":::::BaseClassificationTask:::::")
        # print(f"=====> {Task=}")
        # print(f"=====> {target=} ")
        # print(f"=====> {prediction=}")
        assert (prediction.dim() == 2) & (target.shape[0] == prediction.shape[0]), f'{prediction.shape=}, {target.shape=}'
        if target.dim() == 1:
            # assume targets are classes
            return F.cross_entropy(prediction, target)
        elif target.dim() == 2:
            # assume targets are probabilities 
            return self.cross_entropy_loss(target, prediction)

    def metrics(self, prediction, target):
        labels = target if target.dim() == 1 else target.argmax(1)
        # TODO: change to bar plot/table
        p_classes = prediction.argmax(1)
        rates = {
            f'rate_{c}': (p_classes == c).float().mean().detach() for c in range(prediction.shape[1])
        }

        return {
            'cross_entropy': self.loss(prediction, target).item(),
            'acc': (prediction.argmax(1) == labels).float().mean().item(),
            **rates,
        }

    @staticmethod
    def cross_entropy_loss(target, q):
        assert target.dim() == 2 and q.dim() == 2
        loss = -(F.softmax(target, 1) * F.log_softmax(q, 1)).sum(1)
        # loss = (F.softmax(p, 1) * F.log_softmax(q, 1)).sum(1) + (F.softmax(p, 1) * F.log_softmax(q, 1)).sum(1)
        return loss.mean()
    

class CIFARClassificationTask(BaseClassificationTask):
    N = 50000   #  50000 training images and 10000 test images.
    DIM = 2

    def __init__(
        self,
        task_type: str ='random',
        task_idx: int = 0,
        net_arch: str = 'resnet18',
        dataset: str = 'cifar10',
        n_classes: int = 2,
    ):
        super().__init__()

        self.task_type = task_type
        if dataset == 'tiny_imagenet':
            self.N = 120000
            
        self.DIM = n_classes
            
        if self.task_type == 'random':
            g = torch.Generator().manual_seed(task_idx)
            table = torch.randint(0, self.DIM, (self.N,), generator=g)
            print(f'[TASK] ===> Random task: {table[:20]}')
        elif self.task_type == 'real':
            assert self.DIM in [2, 10], f'{self.DIM}-way classification tasks are NOT supported'
            if dataset == 'tiny_imagenet' or dataset == 'tiny_imagenet_64':
                label2val = [int(i in TINY_IMAGENET_REAL_BIN_TASKS[task_idx]) for i in range(200)]
                # label2val = [int(i < 100) for i in range(200)]
                table = torch.LongTensor(label2val)
            elif dataset == 'cifar10':
                if self.DIM == 2:
                    label2val = [int(i in CIFAR_REAL_BIN_TASKS[task_idx]) for i in range(10)]
                    table = torch.LongTensor(label2val)
                elif self.DIM == 10:
                    table = torch.arange(0, 10).long()
            print(f'[TASK] ===> Real task: {table}')
        elif self.task_type.startswith('net'):
            if net_arch == 'resnet18':
                self.task_net = ResNet18(out_dim=self.DIM)
            else:
                raise NotImplementedError
            table = torch.zeros(self.N,).long()
        elif self.task_type == 'table':
            table = torch.zeros(self.N).long()
            print(f'[TASK] ===> Table task: {table[:20]}')

        elif self.task_type.startswith('net'):
            if net_arch == 'resnet18':
                self.task_net = ResNet18(out_dim=self.DIM)
            else:
                raise NotImplementedError
            table = torch.zeros(self.N,).long()
        elif self.task_type == 'table':
            table = torch.zeros(self.N).long()
            print(f'[TASK] ===> Table task: {table[:20]}')

        self.lookup_table = nn.parameter.Parameter(table, requires_grad=False)

    def forward(self, x=None, y=None, idx=None):
        if self.task_type in ["random", 'table']:
            assert idx is not None
            t = self.lookup_table[idx]
        elif self.task_type == "real" or self.task_type == "real10":
            assert y is not None
            t = self.lookup_table[y]
        elif self.task_type.startswith("net"):
            assert x is not None
            t = self.task_net(x)
            if not self.task_type.endswith('logits'):
                t = t.argmax(1)
        else:
            raise NotImplementedError

        return t

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        if self.task_type == 'net':
            res = self.task_net.load_state_dict(state_dict, strict=strict)
        else:
            res = super().load_state_dict(state_dict, strict=strict)

        print(f'[TASK] ===> Loaded "{self.task_type}" task: example={self.lookup_table[:20].tolist()}, mean={self.lookup_table.float().mean():.2f}')
        return res

class AudiosetClassificationTask(BaseClassificationTask):
    N = 39731
    DIM = 2

    def __init__(
            self,
            task_type: str='random',
            task_idx: int = 0,  # TODO: display name to idx
            net_arch: str = 'resnet18',
            dataset: str = 'audioset',
            n_classes: int = 2,
    ):
        super().__init__()

        self.task_type = task_type
        self.DIM = n_classes
        self.dataset = dataset

        if self.dataset  == 'audioset':
            self.N=39731
        # elif self.dataset == 'audiosetpop':
        #     self.N=499
        # elif self.dataset == 'audiosetclassical':
        #     self.N=2551
        elif self.dataset == 'audiosetoutside_urban_or_manmade':
            self.N=12292     # baseline: 12292, ours (vs_changepoint): 65665
        elif self.dataset == 'audiosetoutside_rural_or_natural':
            self.N=12463     # baseline: 12463, ours (vs_changepoint): 67298
        elif self.dataset == 'audiosetdomestic_sounds_home_sounds':
            self.N = 8237   # baseline: 8237, ours (vs_changepoint): 43218
        elif self.dataset == 'audiosetspeech':
            self.N = 16704
        elif self.dataset == 'audiosetspeech_single_label':
            self.N = 4452   # baseline: 7xx, ours (vs_changepoint): 4452
        else:
            raise NotImplementedError

        def generate_random_task_id_table(task_idx):
            g = torch.Generator().manual_seed(task_idx)
            table = torch.randint(0, self.DIM, (self.N,), generator=g)
            print(f'[TASK] ===> Random task: {table[:20]}')
            return table

        if self.task_type == 'random':
            table = generate_random_task_id_table(task_idx)

        elif self.task_type == 'real':
            assert self.DIM in [2, 50], f'{self.DIM}-way classification tasks are NOT supported'
            if dataset == 'original':   # TODO
                warnings.warn(f"[AudiosetClassificationTask] ===> original is not implemented yet", UserWarning)
                sys.exit()
            elif dataset in ['audiosetoutside_urban_or_manmade',
                             'audiosetoutside_rural_or_natural', 'audiosetdomestic_sounds_home_sounds',
                             'audiosetspeech', 'audiosetspeech_single_label']:
                include_classes_dict = {'audiosetdomestic_sounds_home_sounds': set(range(25)),
                                        'audiosetoutside_urban_or_manmade': set(range(25)), # 127
                                        'audiosetoutside_rural_or_natural': set(range(25)), # 154
                                        'audiosetspeech': set(range(4)),
                                        'audiosetspeech_single_label': set(range(4))}
                num_include_classes_dict = {'audiosetdomestic_sounds_home_sounds': 25,
                                        'audiosetoutside_urban_or_manmade': 25,    # 127
                                        'audiosetoutside_rural_or_natural': 25,    # 154
                                        'audiosetspeech': 4,
                                        'audiosetspeech_single_label': 4}
                if self.DIM == 2:
                    AUDIOSET_REAL_BIN_TASKS = list(generate_all_binary_tasks(include_classes_dict[dataset]))
                    label2val = [int(i in AUDIOSET_REAL_BIN_TASKS[task_idx]) for i in range(num_include_classes_dict[dataset])]
                    table = torch.LongTensor(label2val)
                    #table = generate_random_task_id_table(task_idx)
                elif self.DIM == num_include_classes_dict[dataset]:
                    table = torch.arange(0, num_include_classes_dict[dataset]).long()
                print(f'[TASK] ===> Real task: {table}')
        elif self.task_type.startswith('net'):
            if net_arch == 'resnet18':
                self.task_net = ResNet18(out_dim=self.DIM)
            elif net_arch == 'resnet8':
                self.task_net = ResNet8(out_dim=self.DIM)
            elif net_arch == 'resnet50':
                self.task_net = ResNet50(out_dim=self.DIM)
            else:
                raise NotImplementedError
            table = torch.zeros(self.N, ).long()
        elif self.task_type == 'table':
            table = torch.zeros(self.N).long()
            print(f'[TASK] ===> Table task: {table[:20]}')

        elif self.task_type.startswith('net'):
            if net_arch == 'resnet18':
                self.task_net = ResNet18(out_dim=self.DIM)
            elif net_arch == 'resnet8':
                self.task_net = ResNet8(out_dim=self.DIM)
            elif net_arch == 'resnet50':
                self.task_net = ResNet50(out_dim=self.DIM)
            else:
                raise NotImplementedError
            table = torch.zeros(self.N, ).long()
        elif self.task_type == 'table':
            table = torch.zeros(self.N).long()
            print(f'[TASK] ===> Table task: {table[:20]}')

        self.lookup_table = nn.parameter.Parameter(table, requires_grad=False)

    def forward(self, x=None, y=None, idx=None):
        if self.task_type in ["random", 'table']:
            assert idx is not None
            t = self.lookup_table[idx]
        elif self.task_type == "real" or self.task_type == "real10":
            assert y is not None
            t = self.lookup_table[y]
        elif self.task_type.startswith("net"):
            assert x is not None
            t = self.task_net(x)
            if not self.task_type.endswith('logits'):
                t = t.argmax(1)
        else:
            raise NotImplementedError

        return t

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        if self.task_type == 'net':
            res = self.task_net.load_state_dict(state_dict, strict=strict)
        else:
            res = super().load_state_dict(state_dict, strict=strict)

        print(
            f'[TASK] ===> Loaded "{self.task_type}" task: example={self.lookup_table[:20].tolist()}, mean={self.lookup_table.float().mean():.2f}')
        return res



class CIFAREmbeddingClassificationTask(BaseClassificationTask):
    DIM = 2

    def __init__(
        self,
        h_dim: int,
        in_dim: List[int] = (3,),
        out_type: str = 'logits',
        arch: str = 'resnet18',
        proj: str = 'linear',
        n_classes: int = 2,
    ):  
        super().__init__()
        # self.n_linear_tasks = n_linear_tasks or h_dim
        # self.task_idx = task_idx
        self.out_type = out_type
        self.DIM = n_classes

        self.encoder = TaskDiscoveryEncoder(
            in_dim=in_dim,
            h_dim=h_dim,
            arch=arch,
            proj=proj,
            n_classes=n_classes,
        )

    def forward(self, x=None, y=None, idx=None):
        self.encoder.eval()

        p = self.encoder(x)[0]
        if self.out_type == 'class':
            p = p.argmax(1)
        elif self.out_type == 'logits':
            pass
        else:
            raise ValueError(f'{self.out_type=}')
        return p


class AudiosetEmbeddingClassificationTask(BaseClassificationTask):
    DIM = 2
    def __init__(
            self,
        h_dim: int,
        in_dim: List[int] = (3,),
        out_type: str = 'logits',
        arch: str = 'resnet18',
        proj: str = 'linear',
        n_classes: int = 2,
    ):
        super().__init__()
        self.out_type = out_type
        self.DIM = n_classes

        self.encoder = TaskDiscoveryEncoder(
            in_dim=in_dim,
            h_dim=h_dim,
            arch=arch,
            proj=proj,
            n_classes=n_classes,
        )

    def forward(self, x=None, y=None, idx=None):
        self.encoder.eval()

        p = self.encoder(x)[0]
        if self.out_type == 'class':
            p = p.argmax(1)
        elif self.out_type == 'logits':
            pass
        else:
            raise ValueError(f'{self.out_type=}')
        return p