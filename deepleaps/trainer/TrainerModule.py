from deepleaps.app.app import App, WorksapceUtility
from deepleaps.app.Format import MainStateBasedFormatter
from deepleaps.utils.config import Config
import os
import torch
from deepleaps.utils.config import Config
import copy
from deepleaps.utils.runtime import get_class_object_from_name, get_class_object_from_load_info
from deepleaps.trainer.TrainerContainer import TrainerContainer
import sys
import shutil
from deepleaps.utils.file_struct import *
''':
required:
args['module_name'] = module_object name
args['file_name'] = saved file name, default: module_name.ckp
args['module_info'] = information for modules
'''

class TrainerModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if isinstance(self.config, str):
            self.config = Config.from_yaml(self.config)
        else:
            self.config = copy.deepcopy(self.config)

    def get_name_format(self, controller, identifier):
        f = MainStateBasedFormatter(controller, {'model_name': App.instance().name, 'identifier': identifier}, '[$identifier]_[$model_name]_[$main:step:03]_[$main:total_step:08].ckpt')
        return f.Formatting()

    def load_hook(self, args):
        pass

    def save_hook(self, args):
        ws: WorksapceUtility = args['workspace']
        ws.add_path('file_name', self.get_name_format(args['controller'], args['identifier']))

    # data, module_info, module_load
    def save(self, args):
        workspace: WorksapceUtility = args['workspace']
        ckp_dir = workspace.get_assigned_full_path('ckp_dir')
        App.instance().make_save_dir(ckp_dir)

        saved_data = {'data': self.state_dict()}
        if args.get('use_hook', True):
            args['module_info'] = {}
            self.save_hook(args) # get module info
            if isinstance(args['module_info'], Config):
                args['module_info'] = Config.extraction_dictionary(args['module_info'])
            saved_data['module_info'] = args['module_info']

        script_path = os.path.join(workspace.get_assigned_relative_path('script_dir'), '{}.py'.format(args['module_name']))
        src = sys.modules[self.__class__.__module__].__file__
        dst = os.path.join(workspace.assigned_base_path, script_path)
        try:
            shutil.copy(src, dst)
        except shutil.SameFileError as e:
            pass

        saved_data['module_config']: dict = Config.extraction_dictionary(self.config)
        saved_data['module_config'].get('_LOAD_INFO', {}).update(
            {'_LOAD_TYPE': 'file',
             '_MODULE_NAME': script_path,
             '_reload': True
             })

        if 'file_name' not in workspace.keys():
            workspace.add_path('file_name', args.get('file_name', args['module_name'] + '.ckp'))
        torch.save(saved_data, workspace.get_assigned_full_path('ckp_dir', 'file_name'))
        App.instance().set_variables('$latest_{}'.format(args['module_name']), workspace.get_relative_path('ckp_dir', 'file_name'))

    def load(self, args):
        workspace: WorksapceUtility = args['workspace']
        ckp_path = workspace.get_assigned_full_path('ckp_path')

        state_dict = torch.load(ckp_path)

        if 'load_hook' in dir(self) and args.get('use_hook', True):
            args['module_info'] = state_dict['module_info']
            self.load_hook(args)

        self.load_state_dict(state_dict['data'], strict=args['load_strict'])

    @classmethod
    def insert_trainer_module_from_checkpoint(cls, args):
        workspace: WorksapceUtility = args['workspace']

        state_dict = torch.load(workspace.get_assigned_full_path('ckp_path'))

        config = state_dict['module_config']
        config['_LOAD_INFO']['_WORKSPACE'] = workspace.assigned_base_path
        cls.insert_trainer_module(args['module_name'], config)

    @staticmethod
    def config_load(config):
        if isinstance(config, str):
            config = Config.from_yaml(config)
        else:
            config = copy.deepcopy(config)
        return config

    def update_module(self, config):
        pass

    @classmethod
    def factory(cls, config):
        obj = cls(config)
        obj.update_module(config)
        return obj

    @classmethod
    def insert_trainer_module(cls, name, config):
        if isinstance(config, dict):
            config = Config.from_dict(config)

        container = App.instance().register_buffer['container']
        if config.get('_reload', True) or container.__dict__.get(name, None) is None:
            target = get_class_object_from_load_info(config._LOAD_INFO).factory(
                config)
        else:
            target = App.instance().register_buffer['previous_container'].__getattribute__(name).update_module(config)
        container.__setattr__(name, target)

    @classmethod
    def make_trainer_module(cls, config):
        config = cls.config_load(config)

        App.instance().register_buffer['previous_container'] = App.instance().register_buffer['container']
        App.instance().register_buffer['container'] = get_class_object_from_load_info(config._LOAD_INFO).factory(config)
        container: TrainerContainer = App.instance().register_buffer['container']
        for key in config.keys():
            if key.startswith('_'): continue
            sub_module_config = cls.config_load(config[key])
            cls.insert_trainer_module(key, sub_module_config)
        App.instance().register_buffer['previous_container'] = None
        return container

from collections import defaultdict
if __name__ == '__main__':
    print('a')
    print(os.path.basename('askfa/safklwklf/file_name.tow'))
#    TrainerModule.make_trainer_module('/home/cvip/Documents/leaps_event/deepleaps/workspace/resource/configs/trainer/ExampleContainer.yaml')

'''
            if config.get('_from_checkpoint', False):
                load_info = config._LOAD_INFO
                workspace = WorksapceUtility(load_info._WORKSPACE)
                workspace_meta = get_workspace_meta(workspace.assigned_base_path)
                workspace.add_path('checkpoint_path', load_info._CHECKPOINT_PATH)
                checkpoint_path = workspace.get_assigned_relative_path('checkpoint_path')

                if checkpoint_path.split('.')[-1].startswith('id'):
                    head, tail = os.path.basename(checkpoint_path).split('.')
                    checkpoint_path = get_meta_value(workspace_meta, 'checkpoint', name, head)
                    workspace.add_path('checkpoint_path', checkpoint_path)
                checkpoint_path = workspace.get_assigned_full_path('checkpoint')

                state_dict = torch.load(checkpoint_path)
                module_config = state_dict['module_config']
                config = module_config.update(config)
'''