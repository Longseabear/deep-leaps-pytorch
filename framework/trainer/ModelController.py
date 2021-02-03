from framework.app.app import SingletoneInstance, App
from framework.dataloader.DataLoader import DataLoaderController
from framework.utils.runtime import get_class_object_from_name
from framework.trainer.losses import LossContainer
from framework.ipc.RunningGraphController import RunnableModuleController
from tqdm import tqdm
import torch
import time

class ModelController(SingletoneInstance):
    def __init__(self):
        self.config = App.instance().config.MODEL_CONTROLLER
        self.dataloader_controller = DataLoaderController.instance()

        self.MODEL: torch.nn.Module = None
        self.OPTIMIZER: torch.nn.Module = None
        self.LOSSES: torch.nn.Module = App.instance().set_gpu_device(LossContainer(self.config.LOSSES))

        self.COMMAND_CONTROLLER = RunnableModuleController(self.config.COMMAND_CONTROLLER, self)
        self.sample = None

        self.all_callable = [method_name for method_name in dir(self) if callable(getattr(self, method_name))]

    @classmethod
    def register(cls, config=None):
        if config is None:
            config = App.instance().config.MODEL_CONTROLLER

        controller_module = config.MODULE_NAME
        controller_class = config.CLASS_NAME
        controller_module: ModelController = get_class_object_from_name(controller_module, controller_class)
        return controller_module.instance()
