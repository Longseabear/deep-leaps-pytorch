import matplotlib.pyplot as plt
import copy
from deepleaps.app.app import App, WorksapceUtility
from deepleaps.app.Format import *
from collections import defaultdict
from deepleaps.ipc.ThreadCommand import *
from deepleaps.utils.config import Config
from deepleaps.utils.data_structure import OrderedSet
from deepleaps.dataloader.DataLoader import DataLoaderController
import traceback
from deepleaps.app.Exceptions import *
from tqdm import tqdm
import os
from deepleaps.dataloader.Transform import TRANSFORM
from deepleaps.dataloader.TensorTypes import *
from deepleaps.trainer.TrainerModule import TrainerModule
from deepleaps.utils.file_struct import get_meta_value, get_workspace_meta
import pathlib
import shutil
import sys

class DependentParent(object):
    def __init__(self, my_name, run_cycle: str):
        abb = run_cycle.split(':')
        default_step = 1
        if len(abb) > 1:
            run_cycle = abb[0]
            default_step = int(abb[1])

        cut = run_cycle.split('.', maxsplit=1)
        if len(cut)==1: cut.append("step")

        self.my_name = my_name
        self.dependent_module, self.attribute = cut
        self.requried_step = default_step
        self.previous_step = -1

    def get_dependent_module(self):
        return RunnableModule.get_runnable_controller().get_runnable_module(self.dependent_module)

    def get_condition_value(self):
        return RunnableModule.get_runnable_controller().get_runnable_module(self.dependent_module).__getattribute__(self.attribute)

    def valid_check(self):
        my = RunnableModule.get_runnable_controller().get_runnable_module(self.my_name)
        now_step = self.get_condition_value()
        if now_step % self.requried_step == 0 and self.previous_step != now_step and my.live:
            self.previous_step = now_step
            return my.repeat == -1 or my.repeat > my.step
        return False

class RunnableModule(object):
    __cls_counter = defaultdict(int)
    __runnable_controller = None
    __factory = None
    __global_count = 0

    @classmethod
    def get_default_config(cls, command_name):
        args = {}
        args['command'] = command_name
        args['required'] = []
        args['repeat'] = 1
        args['args'] = {}
        return Config.from_dict(args)

    @property
    def factory(self):
        return RunnableModule.__factory

    def set_step(self, val):
        self.step = val
        self.dependent.previous_step = -1

    @classmethod
    def set_global_module(cls, runnable_controller):
        cls.__runnable_controller = runnable_controller
        cls.__factory = runnable_controller.factory

    @classmethod
    def get_cls_count(cls):
        return cls.__cls_counter

    @classmethod
    def get_trainer_container(cls):
        return App.instance().register_buffer['container']

    @classmethod
    def get_runnable_controller(cls):
        return cls.__runnable_controller

    def log_format(self, contents):
        return contents + " <{}:{}>".format(self.__class__.__name__, self.name)

    def __init__(self, name, config):
        """

        :param name:
        :param config: must to be configuration safe
        :param generation:
        :param parent:
        """
        self.config = config
        self.required = OrderedSet(self.config.get('required', []))

        self.name = name
        self.step = 0
        self.finish = self.config.get('finish', False)
        self.live = True

        # Graph Module
        self.indegree = 0
        self.current_degree = 0

        App.instance().logger.info(self.log_format("COMMAND JOIN"))
        """
        Command Class
        Command class can be used with the with statement.
        :param name:
        :param config:
        """
        self.dependent = DependentParent(self.name, self.config.get('run_cycle', "$self.step:1"))
        self.repeat = int(self.config.get('repeat', 1, False))
        if not 'args' in self.config.keys():
            self.config.args = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.current_degree = 0

    def __enter__(self):
        return self

    def start(self):
        """
        Must be executed before the module is executed.
        """
        RunnableModule.get_runnable_controller().variables['self'].append(self.name)

    def valid(self):
        return self.dependent.valid_check()

    def run(self):
        """
        Must be executed before the module is executed.
        """
        raise NotImplementedError

    def __del__(self):
        App.instance().logger.info(self.log_format("COMMAND EXPIRED"))

    def update(self):
        """
        Must be executed before the module is executed.
        """
        raise NotImplementedError

    def end(self):
        self.current_degree = 0

        if RunnableModule.get_runnable_controller().variables['self'][-1] == self.name:
            RunnableModule.get_runnable_controller().variables['self'].pop()

        if self.finish and (self.repeat != -1 and self.repeat <= self.step):
            self.live = False

    def leave(self):
        return self.live

    def destroy(self):
        raise NotImplementedError

class RunnableGraph(RunnableModule):
    def __init__(self, name, config, **kwargs):
        super().__init__(name, config)
        self.process_queue = queue.Queue()

    def start(self):
        super(RunnableGraph, self).start()

    def end(self):
        super(RunnableGraph, self).end()

    def init(self):
        self.process_queue = queue.Queue()
        self.step += 1

        # UPDATE GRAPH
        self.factory.update_graph()
        for obj_name in self.required:
            obj = RunnableModule.get_runnable_controller().get_runnable_module(obj_name)
            obj.step = 0
            obj.current_degree = 0
            if obj.indegree == 0:
                self.process_queue.put(obj_name)

    def update(self):
        pass

    def destroy(self):
        module_controller = RunnableModule.get_runnable_controller()
        for obj_name in self.required:
            module_controller.get_runnable_module(obj_name).destroy()
        module_controller.remove_runnable_module(self.name)

    def remove_edge(self, name):
        module_controller = RunnableModule.get_runnable_controller()
        next_names = module_controller.next[name]
        previous_names = module_controller.previous[name]

        for next_name in next_names:
            next_obj = module_controller.get_runnable_module(next_name)
            next_obj.indegree -= 1
            next_obj.indegree += len(previous_names)
            next_obj.current_degree += len(previous_names)

            module_controller.previous[next_name].remove(name)
            module_controller.previous[next_name] += previous_names
            if next_obj.current_degree == next_obj.indegree:
                self.process_queue.put(next_name)

        for pname in previous_names:
            module_controller.next[pname].remove(name)
            module_controller.next[pname] += next_names

        del module_controller.next[name]
        del module_controller.previous[name]
        self.required.remove(name)

    def process(self, obj_name):
        module_controller = RunnableModule.get_runnable_controller()
        try:
            with module_controller.get_runnable_module(obj_name) as p:
                if p.valid():
                    p.start()
                    p.run()
        except MainGraphStepInterrupt as e:
            if isinstance(self, MainGraph):
                pass
            else:
                e.graph_exception_processing(self, p)
        except MainGraphFinishedException as e:
            e.graph_exception_processing(self, p)
        except Exception as e:
            if App.instance().config.App.DEBUG:
                print(traceback.format_exc())
            else:
                App.instance().logger.warning('[ERROR/{}/{}] {}'.format(p.name, p.__class__.__name__, e))
        p.update()
        p.end()
        if not p.leave():
            self.remove_edge(p.name)
            p.destroy()
        else:
            nexts_name = module_controller.next[p.name]
            for name in nexts_name:
                next_obj = module_controller.get_runnable_module(name)
                next_obj.current_degree += 1
                if next_obj.current_degree == next_obj.indegree:
                    self.process_queue.put(name)

    # Must to SCC
    def run(self):
        while self.valid():
            self.init()
            while not self.process_queue.empty():
                obj_name: str = self.process_queue.get()
                self.process(obj_name)

class MainGraph(RunnableGraph):
    def __init__(self, name, config, **kwargs):
        super().__init__(name, config, **kwargs)
        self.iter = None
        self.tqdm = None

        self.total_step = 0
        self.loader_name = self.config.args.loader_name

    def get_epoch(self):
        return self.step - 1

    def start(self):
        super(MainGraph, self).start()
        RunnableModule.get_runnable_controller().variables['main'].append(self.name)

        config = copy.deepcopy(self.config.args[self.loader_name]) if self.loader_name in self.config.args.keys() else None
        DataLoaderController.instance().make_dataset(self.loader_name, config)
    
    def init(self):
        super(MainGraph, self).init()
        self.tqdm = tqdm(DataLoaderController.instance().dataloaders[self.loader_name],
                                       bar_format='{l_bar}{bar:10}{r_bar}', ascii=True)
        self.tqdm.desc = '[{}] {}/{} '.format(self.loader_name, self.step, self.repeat)
        self.iter = iter(self.tqdm)

    def destroy(self):
        super(MainGraph, self).destroy()
        self.tqdm = None
        self.iter = None

    def end(self):
        super(MainGraph, self).end()
        if RunnableModule.get_runnable_controller().variables['main'][-1] == self.name:
            RunnableModule.get_runnable_controller().variables['main'].pop()

    def run(self):
        try:
            super(MainGraph, self).run()
        except MainGraphFinishedException as e:
            App.instance().logger.warning(self.log_format('Intrrupt occur: {}'.format(e)))

# -------------------------------- NODE
class RunnableNode(RunnableModule):
    def __init__(self, name, config):
        super().__init__(name, config)

    def start(self):
        pass

    def run(self):
        raise NotImplementedError

    def update(self):
        self.step += 1

    def end(self):
        super(RunnableNode, self).end()

    def destroy(self):
        module_controller = RunnableModule.get_runnable_controller()
        module_controller.remove_runnable_module(self.name)

class TestCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        new_step = int(self.config.args.new_step)
        obj = self.get_runnable_controller().get_current_main_module()
        previous_step = obj.step
        obj.step = new_step
        App.instance().logger.info(
            self.log_format('previous step: {} -> new step {}'.format(previous_step, new_step)))
        if not obj.valid():
            raise MainGraphFinishedException

class PrintCommand(RunnableNode):
    def __init__(self, name, config):
        config = Config.from_dict({
            'args':{
                'content': 'default contents'
            }
        }).update(config)
        super().__init__(name, config)

    def run(self):
        str = '{} step: {} / repeat: {}'.format(self.config.args.content, self.dependent.get_dependent_module().__getattribute__(self.dependent.attribute), self.repeat) + " "
        for name in self.required:
            str += name + " "
        App.instance().logger.info(self.log_format(str))

class PrintStateCommand(RunnableNode):
    def __init__(self, name, config):
        config = Config.from_dict({
            'args':{
                'obj': '$main'
            }
        }).update(config)
        super().__init__(name, config)

    def run(self):
        obj = self.get_runnable_controller().get_runnable_module(self.config.args.obj)
        str = ''
        for name in self.required:
            str += '[{}.{}]: {} / '.format(obj.name, name, obj.__getattribute__(name)) + ' '
        App.instance().logger.info(self.log_format(str))

class RunCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        for name in self.required:
            if name in dir(self.get_trainer_container()):
                self.get_trainer_container().__getattribute__(name)(**self.config.args)

class BatchedImageShowCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.numpy_trasnsform = TRANSFORM['ToNumpy']([name + "_output" for name in self.config.required], {'ALL': IMAGE()})

    def run(self):
        temp_sample = {}
        for name in self.config.required:
            temp_sample[name + "_output"] = self.get_trainer_container().sample[name]
        temp_sample = self.numpy_trasnsform(temp_sample)

        for key in temp_sample.keys():
            plt.imshow(temp_sample[key][self.config.args.batch_number])
            plt.show()

class BatchedImageSaveCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.numpy_trasnsform = TRANSFORM['ToNumpy']([name for name in self.config.required], {'ALL': IMAGE()})
        self.type = IMAGE()

    def leave(self):
        if self.live or not MultipleProcessorController.instance().finished(self.__class__.__name__):
            return True
        else:
            return False

    def destroy(self):
        MultipleProcessorController.instance().remove_process(self.__class__.__name__)

    def run(self):
        args = []
        ws = WorksapceUtility(self.config.args.get('base_path', '$base'))
        ws.add_path('dir_path', self.config.args.get('path', 'visual/img', possible_none=False))
        fm = self.config.args.get('format', 'png', possible_none=False)

        App.instance().make_save_dir(ws.get_assigned_full_path('dir_path'))
        formatter = MainStateBasedFormatter(self.get_runnable_controller(), {'content': '', 'format': fm, 'batch':0},
                              format='[$main:step:03]e_[$main:total_step:08]s_[$content]_[$batch].[$format]')

        for name in self.config.required:
            formatter.contents['content'] = name
            imgs = self.numpy_trasnsform({name:self.get_trainer_container().sample[name].clone()})

            b,_,_,_ = imgs[name].shape
            for i in range(b):
                formatter.contents['batch'] = str(i).zfill(4)
                path = os.path.join(ws.get_assigned_full_path('dir_path'), formatter.Formatting())
                args.append((path, imgs[name][i]))

        import time
        def batched_image_save(queue):
            while True:
                sample = queue.get()
                if sample is None: break
                path, img = sample
                misc.imsave(path, img)
                time.sleep(0.001)
        MultipleProcessorController.instance().push_data(self.__class__.__name__, batched_image_save, args, num_worker=1)


class ModuleLoadClass(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        workspace = WorksapceUtility(self.config.get('base_path', '$base'))
        # checkpoint / name / identifier / relative path
        workspace_meta = get_workspace_meta(workspace.assigned_base_path)

        for name in self.required:
            args = Config.from_dict({
                'module_name': name,
                'controller': self.get_runnable_controller(),
                'workspace': workspace,
                'load_strict': True,
                'ckp_path': '$latest.id'
            })
            args.update(self.config.args.get(name, {}))
            # $base/121212-020202.identifier
            try:
                ckp_path: str = args['ckp_path']
                if ckp_path == '$latest':
                    ckp_path = '$latest_{}'.format(name)

                if pathlib.Path(ckp_path).suffix.startswith('.id'):
                    head, tail = os.path.basename(ckp_path).split('.')
                    ckp_path = get_meta_value(workspace_meta, 'checkpoint', name, head)

                if ckp_path == None:
                    App.instance().logger.error(self.log_format('Load fail: there is no identifier file. {}'.format(head)))
                    continue

                workspace.add_path('ckp_path', ckp_path)

            except KeyError as e:
                App.instance().logger.error(self.log_format('App.variables load error. key:{}.'.format(args['ckp_path'])))
                continue

            '''
            If trainer_module_load is true or no trainer is registered,
             a new module is loaded first.
            '''
            if name not in dir(self.get_trainer_container()) or args.get('_reload', True):
                TrainerModule.insert_trainer_module_from_checkpoint(args)
                App.instance().logger.info(self.log_format('RELOAD MODULE: {}'.format(name)))

            module = self.get_trainer_container().__getattribute__(name)
            module.load(args)
            App.instance().logger.info(self.log_format('LOAD Success {:>12.12} [{}] '.format(name, workspace.get_assigned_full_path('ckp_path'))))

class ModuleSaveClass(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        identifier = App.instance().time_format()
        workspace = WorksapceUtility(self.config.get('base_path', '$base'))
        workspace.add_path('script_dir', self.config.get('script_dir', 'script'))

        App.instance().make_save_dir(workspace.get_assigned_full_path('script_dir'))

        for name in self.required:
            if name not in dir(self.get_trainer_container()):
                App.instance().logger.error(self.log_format(
                    'save fail: there is no module: {}'.format(name)))
                continue
            args = Config.from_dict({
                'module_name': name,
                'workspace': workspace,
                'controller': self.get_runnable_controller(),
                'identifier': identifier
            })
            args.update(self.config.args.get(name, {}))
            workspace.add_path('ckp_dir', self.config.get('ckp_dir', 'ckpt_{}'.format(name)))

            module = self.get_trainer_container().__getattribute__(name)
            module.save(args)

            App.instance().logger.info(self.log_format(
                'SAVE Success {:>12.12} [{}] '.format(name, App.instance().get_variables('$latest_{}'.format(name)))
            ))

            base = workspace.assigned_base_path
            with open(os.path.join(base, 'ckp.meta'), 'a') as f:
                # checkpoint / name / identifier / relative path
                f.write('{} {} {} {}\n'.format('checkpoint', name, identifier, App.instance().get_variables('$latest_{}'.format(name))))

class TrainerContainerLoaderCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        TrainerModule.make_trainer_module(self.config.args)
