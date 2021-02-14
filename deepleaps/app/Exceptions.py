class MainGraphFinishedException(Exception):
    def __init__(self):
        super().__init__("MainGraph Finished")
    value = property(lambda self: object(), lambda self, v: None, lambda self: None)

    def graph_exception_processing(self, parent, process):
        process.update()
        process.end()
        if not process.leave():
            parent.remove_edge(process.name)
            process.destroy()
        raise MainGraphFinishedException

class MainGraphStepInterrupt(Exception):
    def __init__(self):
        super().__init__()
    value = property(lambda self: object(), lambda self, v: None, lambda self: None)

    def graph_exception_processing(self, parent, process):
        process.update()
        process.end()
        if not process.leave():
            parent.remove_edge(process.name)
            process.destroy()
        raise MainGraphStepInterrupt

class RunnableModuleDestroyException(Exception):
    def __init__(self):
        super().__init__("This runnable module was expired.")

