from importlib import import_module, reload
import importlib.util
from deepleaps.app.app import App
import sys
import torch
import os


def get_instance_from_name(module_path, class_name, *args, **kwargs):
    m = import_module(module_path)
    return getattr(m, class_name)(*args, **kwargs)

def get_class_object_from_file(path, class_name):
    spec = importlib.util.spec_from_file_location(path.replace('/', '.'), path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m.__getattribute__(class_name)

def get_class_object_from_load_info(load_info):
    load_info._MODULE_NAME = App.instance().apply_variable_to_string(load_info._MODULE_NAME)
    load_info._CLASS_NAME = App.instance().apply_variable_to_string(load_info._CLASS_NAME)
    if load_info._LOAD_TYPE == 'file':
        file_path = App.instance().nested_dir_path_parser(os.path.join(load_info._WORKSPACE, load_info._MODULE_NAME))
        return get_class_object_from_file(file_path, load_info._CLASS_NAME)
    return get_class_object_from_name(load_info._MODULE_NAME, load_info._CLASS_NAME)

def get_class_object_from_name(module_path, class_name):
    m = import_module(module_path)
    try:
        return getattr(m, class_name)
    except AttributeError:
        reload(load_module(module_path))
        return getattr(m, class_name)

def load_module(module):
    # module_path = "mypackage.%s" % module
    module_path = module
    if module_path in sys.modules:
        return sys.modules[module_path]
    return __import__(module_path, fromlist=[module])
