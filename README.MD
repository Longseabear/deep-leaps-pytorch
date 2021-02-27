deep-leaps is a deep learning training framework written based on DDD (data driven development). It has the following features.

- **Training/testing procedure abstracted by graph**

  the training & testing procedure is based on graphs. then, you can add or remove nodes in the graph at runtime. For example, if you want to visualize the model in the middle of training, you can add a visualization command at runtime. this does not affect training.

- **Instruction modification at runtime**
  If you want to modify the data or force the learning rate to be adjusted, you can modify the instruction using ipc at runtime.

- **Code modification at runtime**
  deep-leaps allows code modification at runtime. you can command the python file recompile command at runtime using ipc.



---

### Install

```
pip install deep-leaps
```

---

### make workspace

Run the following python command in the project root.

```python
from deepleaps.app.app import App
App.make_workspace('./')
```

result:

```
|--exampleDataset
|    |--0001.jpg
|    |--0002.jpg
|    |--0000.jpg
|--resource
|    |--output
|        |--README.md
|    |--configs
|        |--dataloader
|            |--exampleDataLoader.yaml
|        |--model
|            |--SimpleLayerModel.yaml
|        |--dataset
|            |--Example.yaml
|        |--command
|            |--default_runnable_command.yaml
|        |--trainer
|            |--ExampleContainer.yaml
|        |--default.yaml
|--src
|    |--dataloader
|        |--Exampledataset.py
|        |--TensorTypes.py
|        |--transforms.py
|    |--model
|        |--SimpleConv.py
|    |--trainer
|        |--ExampleContainer.py
|    |--ipc
|        |--CustomCommand.py
|--client.py
|--main.p
```



---

### Command

Unlike the existing framework, deep-leaps is executed depending on the command. 
the basic training process can be applied as follows. please check the example project for more details.

```yaml
DEFAULT: [$root( $TRAINER_CONTAINER_LOAD->$TRAINING_LOADER( $MAIN_MODEL_LOAD->$TRAINING_BATCH( $TRAINING_STEP )->MAIN_MODEL_SAVE) )]

DEFINE:
  TRAINER_CONTAINER_LOAD:
    command: 'TrainerContainerLoaderCommand'
    args: 'resource/configs/trainer/ExampleContainer.yaml'
    finish: True

  MAIN_MODEL_LOAD:
    command: 'ModuleLoadClass'
    required: ['MODEL', 'OPTIMIZER']
    base_path: '$base'
    finish: True
    args:
      MODEL:
        _reload: False
        use_hook: True
        file_name: $latest.id
```

---

### Client

```yaml
COMMAND_CONTROLLER:
  ipc_host: '127.0.0.1'
  ipc_port: 1568
```

You can specify ipc_host and ipc_port through configuration. Using this address, you can communicate with the training network model. the communication example is as follows.

```python
import socket
import threading
from struct import pack

host = '127.0.0.1'
port = 1568
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

try:
    while True:
        data = input(':')
        length = pack('>Q', len(data))
        client_socket.sendall(length)
        client_socket.sendall(data.encode())
        ack = client_socket.recv(1)
        print('send ack ok')
except Exception as e:
    print(e)
    client_socket.close()
```

