from deepleaps.utils.config import Config
from deepleaps.utils.file_struct import copytree
import numpy as np
import logging
import logging.config
import datetime
import sys
import copy
import torch
from tqdm import tqdm
import os
from pathlib import Path
from collections import defaultdict
import pathlib

DEFAULT_CONFIG_PATH = "{}/default_config.yaml".format(os.path.dirname(os.path.abspath( __file__ )))
class SingletoneInstance:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def register(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance # Override
        return cls.__instance

    @classmethod
    def instance(cls):
        cls.__instance = cls()
        cls.instance = cls.__getInstance # Override
        return cls.__instance

    @classmethod
    def reinitialize(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)

    @classmethod
    def assign_instance(cls, instance):
        cls.__instance = instance

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.formatter = logging.Formatter("[%(levelname)s] %(message)s [%(filename)s:%(lineno)d]")

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class App(SingletoneInstance):
    @classmethod
    def register_from_config_list(cls, config_paths):
        """
        Initialize the App by using the configuration list. This function must be executed only once.
        :param config_paths:
        :return: App Class
        """
        config = Config.get_empty()

        for path in config_paths:
            config.update(config.from_yaml(path))
        return cls.register(config, update=True)

    def __init__(self, config=None, update=False):
        """
        App class [Configuration unsafe]

        App is responsible for the overall structure of this deepleaps.
        App has all configurations. please check the list.
        - App
        - MODEL_CONTROLLER:
            - COMMAND_CONTROLLER
            - MODEL_STATE_CONTROLLER
            - OPTIMIZER
            - LOSSES
        - DATA_LOADER
        - SYSTEM

        App can manage environment variables through variables.
        When using variables, use $variable_name.

        :param config:
        :param update: If update is true, all System parameters are executed. (Logger, Seed...)
        """
        self.config = Config.from_yaml(DEFAULT_CONFIG_PATH)
        if config is not None:
            self.config.update(config)

        # --------------------------------SYSTEM--------------------------------------
        self.system = self.config.SYSTEM
        logging.config.dictConfig(Config.extraction_dictionary(self.system.LOGGER))
        self.logger = logging.getLogger('DEFAULT')
        self.logger.addHandler(TqdmLoggingHandler())

        self.setSeed(self.system.SEED)
        # --------------------------------Variable Setting--------------------------------------
        self.name = None
        self._variables = self.config.App.Variables
        self.register_buffer = defaultdict(lambda: None)

    def container_end(self):
        self.name = None
        self.set_variables('$base', None)
        if 'pre' in self._variables.keys():
            self.set_variables('$pre', None)
        self.register_buffer['container'] = None

    def container_join(self, model_name, experiment_name, mode='new'):
        self.name = model_name
        base = self._variables.directory_root
        mode: str  # new, finetuning, resume if mode is finetuning, variables.pre is previous base root

        if mode == 'finetuning':
            if os.path.exists(os.path.join(base, model_name, experiment_name)):
                self._variables.pre = os.path.join(base, model_name, experiment_name)
                experiment_name = self.name_format(experiment_name)
            base_path = os.path.join(base, model_name, experiment_name)
            self._variables.base = base_path
            if os.path.exists(os.path.join(base, model_name, experiment_name)):
                print('[INFO] Now App mode is {}, Since the directory already exists, a new folder {} is created.'.format(mode,
                                                                                                                          self._variables.base))
        elif mode == 'new':
            if os.path.exists(os.path.join(base, model_name, experiment_name)):
                experiment_name = self.name_format(experiment_name)
            base_path = os.path.join(base, model_name, experiment_name)
            self._variables.base = base_path
            if os.path.exists(os.path.join(base, model_name, experiment_name)):
                print('[INFO] Now App mode is {}, Since the directory already exists, a new folder {} is created.'.format(
                    mode, self._variables.base))
        elif mode == 'resume':
            if not os.path.exists(self._variables.base):
                raise FileExistsError
        else:
            raise NotImplementedError
        print('[INFO] base_path: {}'.format(self._variables.base))

    def setSeed(self, seed):
        """
        set seed..
        :param seed:
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.system.REPRODUCIBILITY:
            self.logger.info('REPRODUCIBILITY MODE TRUE')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def name_format(self, name: str)-> str:
        """
        :param name:
        :return: Returns the combined names of common time formats
        """
        now = datetime.datetime.now()
        return now.strftime('{}_%y%m%d_%Hh%Mm'.format(name))

    def time_format(self) -> object:
        """
        :return: Returns common time formats
        """
        now = datetime.datetime.now()
        return now.strftime('%y%m%d-%H%M%S')

    def get_device(self):
        """
        :return: Returns device name
        """
        device_name = self.config.App.device
        return device_name

    def get_gpu_ids(self)->list:
        """
        Return gpu ids
        :return: gpu idxes
        """
        gpu_ids = self.config.App.gpu_ids
        return gpu_ids

    def set_gpu_device(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Set the device in the module. Currently, only the Data parallel method is supported.

        :param module:
        :return:
        """
        device_name = self.config.App.device
        gpu_ids = self.config.App.gpu_ids
        parallel = self.config.App.parallel

        device = torch.device('cpu' if device_name is 'cpu' else 'cuda')

        if device_name is not 'cpu' and len(gpu_ids) > 1 and parallel is "data":
            return torch.nn.DataParallel(module, gpu_ids).to(device)
        return module.to(device)

    def get_base_path(self):
        """
        Return the workspace.
        :return:
        """
        return self.get_variables('$base')

    def make_save_dir(self, path: str):
        """
        After parsing a variable, convert it to the entire path and create a directory.
        This function affects os (create file)
        :param path:
        """
        os.makedirs(self.nested_dir_path_parser(path), exist_ok=True)

    @staticmethod
    def nested_dir_path_parser(path):
        path = ('/' if path.startswith('/') else '') + os.path.join(*App.instance().variable_parsing(path, '/'))
        if '$' in path:
            path = App.nested_dir_path_parser(path)
        return path

    @staticmethod
    def dir_path_parser(path):
        return ('/' if path.startswith('/') else '') + os.path.join(*App.instance().variable_parsing(path, '/'))

    def smart_write(self, contents, dst, mode=None):
        """
        tepmly
        :param contents:
        :param dst:
        :param mode:
        """
        if isinstance(dst, str):
            if dst == 'stdout':
                sys.stdout.write(contents)
            else:
                with open(dst, mode) as f:
                    f.write(contents)
        else:
            with open(dst, mode) as f:
                f.write(contents)

    def set_variables(self, variable, value):
        if variable[0] != '$':
            raise Exception('Variable name must to start with $')
        self._variables[variable[1:]] = value

    def get_variables(self, variable):
        if variable[0] != '$':
            raise Exception('Variable name must to start with $')
        return self._variables[variable[1:]]

    def variable_parsing(self, string, sep=' '):
        """
        Assigns all variables in the given string. variable is defined as $.
        This function returns the cloned result. [safe from shallow copy]
        :param string:
        :param sep:
        :return:
        """
        return [copy.deepcopy(self.get_variables(p)) if p[0] == '$' else p for p in string.split(sep) if len(p)>0]

    def apply_variable_to_string(self, script):
        if '$' not in script:
            return script
        candidate = self.extract_all_variable(script)
        for i in candidate:
            idx = script.find(i)
            value = self.get_variables(i)
            script = script[:idx] + self.apply_variable_to_string(value) + script[idx + len(i):]
        return script

    def extract_all_variable(self, script):
        i = 0
        out = []
        while i < len(script):
            if script[i] == '$':
                i += 1
                temp = ''
                while i < len(script):
                    if not (temp + script[i]).isidentifier(): break
                    temp += script[i]
                    i += 1
                out.append('$' + temp)
            else:
                i += 1
        return out

    @classmethod
    def make_workspace(cls, dst):
        path = Path(os.path.dirname(os.path.abspath(__file__)))
        copytree(os.path.join(path.parent, 'workspace'), dst)

class WorksapceUtility(object):
    def __init__(self, base_path):
        self.base_path = base_path
        self.assigned_base_path = App.instance().nested_dir_path_parser(self.base_path)
        self.paths = {}

    def keys(self):
        return self.paths.keys()

    def split_assigned_full_path(self, key):
        return self.assigned_base_path, self.paths[key]['assigned_path']

    def add_path(self, key, path):
        self.paths[key] = {'path': path, 'assigned_path': App.instance().nested_dir_path_parser(path)}

    def get_assigned_relative_path(self, *keys):
        return os.path.join(*[self.get_path_value(i, 'assigned_path') for i in keys])

    def get_relative_path(self, *key):
        return os.path.join(*[self.get_path_value(i, 'path') for i in key])

    def get_path_value(self, key, type):
        return self.paths[key][type]

    def get_assigned_full_path(self, *key):
        return os.path.join(self.assigned_base_path, self.get_assigned_relative_path(*key))

    def join(self, *args):
        paths = []
        print(args)
        for arg in args:
            if '$' in arg:
                paths.append(App.instance().nested_dir_path_parser(arg))
            else:
                paths.append(arg)
        print(paths)
        input()
        return os.path.join(*paths)