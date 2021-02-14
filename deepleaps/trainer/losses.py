import torch
import torch.nn as nn
from abc import abstractmethod
from deepleaps.utils.config import Config
from deepleaps.app.app import App
from deepleaps.trainer.TrainerModule import TrainerModule
import os
import sys

class LossContainer(TrainerModule):
    def loss_factory(self, name, config):
        for module_name in self.loss_modules:
            current_module = sys.modules[module_name]
            if config.loss_type in dir(current_module):
                return getattr(current_module, config.loss_type)(name, config)
        raise AttributeError

    def update_module(self, config):
        for loss_name in config.keys():
            if loss_name.startswith('_'): continue
            if config.get('_reload', False) or loss_name not in self.loss_dict.keys():
                self.loss_dict[loss_name] = self.loss_factory(loss_name, config[loss_name])

    @classmethod
    def factory(cls, config):
        obj = cls(config)
        obj.update_module(config)
        return App.instance().set_gpu_device(obj)

    def __init__(self, config):
        super().__init__(config)

        self.loss_dict = nn.ModuleDict()
        self.loss_modules = [__name__] + config._LOSS_MODULES
        self.total_loss = LossEmpty('TotalLoss')
        self.device = App.instance().get_device()

    def forward(self, sample):
        total_losses = []
        for key in self.loss_dict.keys():
            inputs = []
            for name in self.loss_dict[key].required:
                inputs.append(sample[name].to(self.device))
            loss = self.loss_dict[key](*inputs)
            loss = self.loss_dict[key].weight * loss
            total_losses.append(loss)

        total_loss = sum(total_losses)
        self.total_loss(total_loss)
        return total_loss

    def save(self, args):
        state = info['state']
        contents = App.instance().current_time_format() + " {} epoch, {} step\n".format(state.epoch, state.step)
        contents += "|{}: {}\t".format('Total', str(self.total_loss))
        for key in self.loss_dict.keys():
            contents += "|{}: {}\t".format(key, str(self.loss_dict[key]))
        App.instance().smart_write(contents + '\n', info['path'], 'a+')

    def load(self, args):
        pass

    def get_dir(self, dir_path):
        return os.path.join(dir_path, 'loss.txt')

class BaseLoss(nn.Module):
    def __init__(self, name=None, config=None):
        super(BaseLoss, self).__init__()
        self.config = config

        self.name = name
        self.weight = config.weight
        self.required = config.inputs

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def update_state(self, loss):
        raise NotImplementedError

class L1Loss(BaseLoss):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.loss = nn.L1Loss()

        self.register_buffer('sum', torch.zeros(1))
        self.register_buffer('count', torch.zeros(1))

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        self.update_state(loss)
        return loss

    def update_state(self, loss):
        self.sum += loss.item()
        self.count += 1

    def __str__(self):
        if self.count==0:
            return "0"
        return str(self.sum.item()/self.count.item())

class LossEmpty(BaseLoss):
    def __init__(self, name, config=None):
        super().__init__(name, Config.from_dict({'weight':1, 'inputs':None}))

        self.register_buffer('sum', torch.zeros(1))
        self.register_buffer('count', torch.zeros(1))

    def forward(self, loss):
        self.update_state(loss)

    def update_state(self, loss):
        self.sum += loss.item()
        self.count += 1

    def __str__(self):
        if self.count==0:
            return "0"
        return str(self.sum.item()/self.count.item())

if __name__ == '__main__':
    config = Config.from_yaml('../../reso   urce/configs/trainer/ExampleContainer.yaml')
    print(config.MODEL_CONTROLLER.LOSSES)
    config.MODEL_CONTROLLER.LOSSES['image_loss_2'] = Config.from_dict({
        'inputs': ['OUTPUT','GT'], 'loss_type': 'L1', 'weight': 2
    })
    loss = LossContainer(config.MODEL_CONTROLLER.LOSSES)
    a = torch.tensor([4.0,1.0]).view(1,1,1,2).float()
    b = torch.tensor([7.0,3.0]).view(1, 1, 1, 2).float()
    sample = {'OUTPUT':a, 'GT':b}
    loss(sample)
    a = torch.tensor([4.0,1.0]).view(1,1,1,2).float()
    b = torch.tensor([7.0,4.0]).view(1, 1, 1, 2).float()
    sample = {'OUTPUT':a, 'GT':b}
    print(loss.total_loss)
    loss(sample)
    print(loss.total_loss)
    print(loss.loss_dict['image_loss_l1'])