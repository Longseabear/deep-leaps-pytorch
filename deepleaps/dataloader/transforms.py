import numpy
from deepleaps.dataloader.TensorTypes import *

"""
Basic transform methods
"""
class Transforms(object):
    def __init__(self, required: list, output_name='{%s}'):
        self.required = required
        self.output_name = output_name

    def output_format(self, input_name):
        return self.output_name.replace('{%s}', input_name)

    def apply(self, sample):
        raise NotImplementedError

    def __call__(self, samples):
        if isinstance(samples, dict):
            for key in self.required:
                samples[self.output_format(key)] = self.apply(samples[key])
        elif isinstance(samples, list):
            for i in range(len(samples)):
                samples[i] = self.apply(samples[i])
        else:
            samples = self.apply(samples)
        return samples

class ToTensor(Transforms):
    def __init__(self, required: list, output_name='{%s}'):
        super(ToTensor, self).__init__(required)

    def apply(self, sample):
        return torch.from_numpy(sample.transpose((2, 0, 1)))

class ToNumpy(Transforms):
    def __init__(self, required: list, output_name='{%s}'):
        super(ToNumpy, self).__init__(required, output_name)

    def apply(self, sample):
        return sample.permute((0, 2, 3, 1)).cpu().detach().numpy()
