import torch
from deepleaps.app.app import App
from deepleaps.utils.runtime import get_class_object_from_name
from deepleaps.utils.config import Config
from deepleaps.app.app import App
import os
from deepleaps.app.app import WorksapceUtility
from deepleaps.app.Format import *
from deepleaps.trainer.TrainerModule import TrainerModule

class Optimizer_Factory(TrainerModule):
    def __init__(self, config):
        super().__init__(config)

    def update_module(self, config):
        pass

    @classmethod
    def factory(cls, config):
        container = App.instance().register_buffer['container']
        trainable = []
        for m in config._PARAMETERS:
            if 'parameters' in dir(container.__getattribute__(m)):
                trainable += list(filter(lambda x: x.requires_grad, container.__getattribute__(m).parameters()))
        optimizer_class = get_class_object_from_name(config.optimizer.optimizer_module,
                                                     config.optimizer.optimizer_class)
        optimizer_args = Config.extraction_dictionary(config.optimizer.optimizer_args)

        scheduler_class = get_class_object_from_name(config.scheduler.scheduler_module,
                                                     config.scheduler.scheduler_class, )
        scheduler_args = Config.extraction_dictionary(config.scheduler.scheduler_args)

        class CustomOptimizer(optimizer_class):
            def update_module(self, config):
                pass

            def __init__(self, *args, **kwargs):
                super(CustomOptimizer, self).__init__(*args, **kwargs)
                self.config = None

            def _register_scheduler(self, scheduler_class, **kwargs):
                self.scheduler = scheduler_class(self, **kwargs)

            @staticmethod
            def get_name_format(controller, identifier):
                f = MainStateBasedFormatter(controller,
                                            {'model_name': App.instance().name, 'identifier': identifier},
                                            '[$identifier]_[$model_name]_[$main:step:03]_[$main:total_step:08].opt')
                return f.Formatting()

            def save_hook(self, args):
                ws: WorksapceUtility = args['workspace']
                ws.add_path('file_name', self.get_name_format(args['controller'], args['identifier']))

                args['module_info'] = {'epoch': args['controller'].get_current_main_module().step}

            def load_hook(self, args):
                epoch = args['module_info'].get('epoch', None)
                if epoch is not None:
                    if epoch >= 1:
                        for _ in range(args['controller'].get_current_main_module().get_epoch(), int(epoch)): self.scheduler.step()
                    args['controller'].get_current_main_module().set_step(int(epoch)+1)

            def load_state_dict(self, *args, **kwargs):
                return super().load_state_dict(*args)

            def save(self, args):
                TrainerModule.save(self, args)

            def load(self, args):
                TrainerModule.load(self, args)

            def schedule(self):
                self.scheduler.step()

            def get_lr(self):
                return self.scheduler.get_lr()[0]

            def get_last_epoch(self):
                return self.scheduler.last_epoch

            def set_config(self, config):
                self.config = config
            # def get_save_name(self, state):
            #     return App.instance().name_format(App.instance().name) + "_{}_{}.opt".format(state.epoch, state.step)

        optimizer = CustomOptimizer(trainable, **optimizer_args)
        optimizer._register_scheduler(scheduler_class, **scheduler_args)
        optimizer.set_config(config)
        return optimizer

