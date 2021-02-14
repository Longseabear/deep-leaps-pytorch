import torch
from deepleaps.utils.runtime import get_class_object_from_name
from deepleaps.utils.config import Config
from deepleaps.app.Format import *
from deepleaps.trainer.TrainerModule import TrainerModule
from deepleaps.app.app import WorksapceUtility
import os

class BaseModel(TrainerModule):
    def __init__(self, config):
        super().__init__(config)
        self.inputs_name = []
        self.outputs_name = []

    def get_name_format(self, controller, identifier):
        f = MainStateBasedFormatter(controller, {'model_name': App.instance().name, 'identifier': identifier}, '[$identifier]_[$model_name]_[$main:step:03]_[$main:total_step:08].model')
        return f.Formatting()

    def description(self):
        return str(self) + "=> IN:{}, OUT:{} ".format(
            self.inputs_name, self.outputs_name
        ) + "device: {}:{}\n".format(App.instance().get_device(), App.instance().get_gpu_ids())

    def update_module(self, config):
        pass

    @classmethod
    def factory(cls, config):
        obj = cls(config)
        obj.update_module(config)
        return App.instance().set_gpu_device(obj)

    def forward(self, samples):
        for name in self.inputs_name:
            samples[name] = samples[name].to(device=App.instance().get_device()).float()
        return samples

# remove latest
'''
        file_names = os.listdir(dst)
        previous_file_name = None
        for name in file_names:
            _, file_extension = os.path.splitext(name)
            if file_extension == 'model':
                epoch = name.split('_')[-2]
                if int(epoch) == info['controller'].get_current_main_module():
                    previous_file_name = name
        if previous_file_name is not None:
            os.remove(os.path.join(dst, previous_file_name))
'''