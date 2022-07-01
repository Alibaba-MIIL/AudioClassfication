from __future__ import division
from __future__ import unicode_literals

from typing import Iterable, Optional
import weakref
import copy
import contextlib

import torch
import math
from collections import OrderedDict
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


@torch.jit.script
def update_ema_jit(ema_v: torch.Tensor, model_v: torch.Tensor, decay_per_step: float, model_factor: float):
    ema_v.mul_(decay_per_step).add_(model_factor * model_v.float())

class ModelEma:
    def __init__(self, model, step_mod_factor=5, decay_per_epoch=0.8):
        # make a copy of the model for accumulating moving average of weights
        # steps_per_epoch - number of ema updates per epoch
        # decay_per_epoch - total decay per epoch

        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay_per_epoch = decay_per_epoch
        self.step_mod_factor = step_mod_factor
        self.decay_per_step = []
        self.ema.cuda()

        self.ema_has_module = hasattr(self.ema, 'module')
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def upload_to_gpu(self):
        self.ema.cuda()

    def download_to_cpu(self):
        self.ema.to(device='cpu')

    def set_decay_per_step(self, num_steps_in_epoch):
        # calculating step_mod_factor
        num_ema_steps_in_epoch = float(num_steps_in_epoch) / self.step_mod_factor
        self.decay_per_step = math.pow(self.decay_per_epoch, 1.0 / num_ema_steps_in_epoch)
        # logger.info("self.decay_per_step {}".format(self.decay_per_step))

    def update(self, model, step):

        # do every step_mod_factor steps
        if step % self.step_mod_factor != 0:
            return

        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k

                # actual update
                if msd[k].dtype == torch.long:  # type Long
                    ema_v.copy_(msd[k])
                else:
                    update_ema_jit(ema_v, msd[k], self.decay_per_step, 1. - self.decay_per_step)

    def get_dict(self, model):
        with torch.no_grad():
            needs_module = hasattr(model, 'module') and not self.ema_has_module
            if not needs_module:
                new_state_dict = deepcopy(self.ema.state_dict())
            else:
                new_state_dict = OrderedDict()
                for k, v in self.ema.state_dict().items():
                    name = 'module.' + k
                    new_state_dict[name] = v
                new_state_dict = deepcopy(new_state_dict)
            return new_state_dict