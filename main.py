import torch
from framework.app.app import App
from framework.dataloader.DataLoader import DataLoaderController
from framework.trainer.ModelController import ModelController
from framework.ipc.ThreadCommand import MultipleProcessorController
import os

print('Device id: ', torch.cuda.current_device())
print('Available: ', torch.cuda.is_available())
print('Property: ', torch.cuda.get_device_properties(0))
print(os.environ['DISPLAY'])

def main(configs):
    if isinstance(configs, list):
        App.register_from_config_list(configs)
    else:
        App.register(configs)
        App.instance().update()
    DataLoaderController.register()
    trainer: ModelController = ModelController.register()
    try:
        trainer.COMMAND_CONTROLLER.run()
    except Exception as e:
        print(e)
    finally:
        MultipleProcessorController.instance().remove_all_process()

if __name__ == '__main__':
    config_list_paths = ['resource/configs/dataloader/dataLoaderEventQueue.yaml',
                         'resource/configs/trainer/ExampleController.yaml']
    main(config_list_paths)