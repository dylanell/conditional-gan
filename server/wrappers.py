"""
Generator model API wrapper class.
"""

import torch
from torchvision.utils import save_image
from typing import List
import sys

# append parent directory
sys.path.append('..')

from conditional_gan.modules import ConditionalGenerator


class GeneratorWrapper():
    def __init__(self, config):
        num_class = config['number_classes']
        input_dim = config['input_dimensions']
        z_dim = config['z_dimension']

        # model artifact file path
        model_file = '../conditional_gan/artifacts/generator.pt'

        # initialize and load model
        self._generator = ConditionalGenerator(z_dim, num_class, input_dim[-1])
        self._generator.load_state_dict(torch.load(
            model_file, map_location=torch.device('cpu')))

        # initialize style vector distribution
        self._style_dist = torch.distributions.normal.Normal(
            torch.zeros(1, z_dim), torch.ones(1, z_dim))

        # set number classes
        self._num_class = num_class

        # sample style distribution to initialize style vector
        self.sample_new_style()

    def get_summary_dict(self):
        return {
            obj.strip(' ').split(':')[0].strip('()'):\
            obj.strip(' ').split(':')[1]\
            for obj in repr(self._generator).split('\n')[1:-1]}

    def sample_new_style(self):
        self._style_vec = self._style_dist.sample()

    def generate_image(self, logits: List[float], media_type='image/png'):
        # create 'multi-hot' distribution from logits list
        label_vec = torch.tensor(logits)
        label_vec /= torch.sum(label_vec)

        # generate sample
        gen_out = self._generator(self._style_vec, label_vec)[0]

        # save generated sample to image file
        if media_type == 'image/png':
            save_image(gen_out, 'gen_out.png')
        else:
            save_image(gen_out, 'gen_out.png')
