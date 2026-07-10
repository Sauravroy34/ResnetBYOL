# The implementation of BYOL is based on https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py

import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from models.auto_aug import autoAUG
from models.resnet_1d import resnet50_1d

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )
    
# main class

class BYOL(nn.Module):
    def __init__(
        self,
        moving_average_decay = 0.99,
        use_momentum = True,
    ):
        super().__init__()

        self.view1 = autoAUG(num_channel=1)
        self.view2 = autoAUG(num_channel=1)

        self.online_encoder = self.create_encoder()
        
        self.online_projector = MLP(dim=2048, projection_size=256, hidden_size=4096)
        
        self.online_predictor = MLP(dim=256, projection_size=256, hidden_size=4096)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_projector = None
        self.target_ema_updater = EMA(moving_average_decay)

    def create_encoder(self):
        encoder = resnet50_1d(in_channels=1, return_last_hidden_state=True)
        return encoder
    
    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder
        
    @singleton('target_projector')
    def _get_target_projector(self):
        target_projector = copy.deepcopy(self.online_projector)
        set_requires_grad(target_projector, False)
        return target_projector

    def reset_moving_average(self):
        del self.target_encoder
        del self.target_projector
        self.target_encoder = None
        self.target_projector = None

    def update_moving_average(self):
        assert self.use_momentum, 'Momentum is turned off for target encoder'
        assert self.target_encoder is not None, 'Target encoder has not been created yet'
        
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        update_moving_average(self.target_ema_updater, self.target_projector, self.online_projector)

    def forward(self, x1, x2):
        view1 = self.view1(x1)
        view2 = self.view2(x2)
        
        online_rep_one = self.online_encoder(view1)
        online_rep_two = self.online_encoder(view2)
        
        online_proj_one = self.online_projector(online_rep_one)
        online_proj_two = self.online_projector(online_rep_two)
        
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_projector = self._get_target_projector() if self.use_momentum else self.online_projector

            target_rep_one = target_encoder(view1)
            target_rep_two = target_encoder(view2)
            
            target_proj_one = target_projector(target_rep_one).detach()
            target_proj_two = target_projector(target_rep_two).detach()

        loss_one = loss_fn(online_pred_one, target_proj_two)
        loss_two = loss_fn(online_pred_two, target_proj_one)

        return (loss_one + loss_two).mean()
