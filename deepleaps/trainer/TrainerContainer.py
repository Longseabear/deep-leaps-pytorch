from deepleaps.app.app import App
from deepleaps.dataloader.DataLoader import DataLoaderController
from deepleaps.utils.runtime import get_class_object_from_name
import copy
from deepleaps.utils.config import Config

class TrainerContainer(object):
    def __init__(self, config):
        self.dataloader_controller = DataLoaderController.instance()
        self.sample = None

#        self.all_callable = [method_name for method_name in dir(self) if callable(getattr(self, method_name))]

    @classmethod
    def set_instance(cls, instance):
        cls.__instance = instance

    def __del__(self):
        App.instance().container_end()

    @classmethod
    def factory(cls, config):
        App.instance().container_join(config._name, config._experiment_name, config.get('_mode', 'new'))
        if App.instance().register_buffer['container'] is None or config.get('_reload', True):
            container = cls(config)
        else:
            container = App.instance().register_buffer['container']
        return container