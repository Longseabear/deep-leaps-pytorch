from deepleaps.dataloader.Transform import TRANSFORM, Transforms
from deepleaps.dataloader.TensorTypes import IMAGE

class ToNumpy(Transforms):
    def __init__(self, required=[], output_name='{%s}'):
        super(ToNumpy, self).__init__(required, output_name)

    def __call__(self, samples):
        for sample in self.required:
            data = samples[sample]
            samples[self.output_format(sample)] = data.permute((0,2,3,1)).cpu().detach().numpy()
        return samples

import inspect, sys
for class_name, module in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if module.__module__ is not __name__: continue
    TRANSFORM[class_name] = module
