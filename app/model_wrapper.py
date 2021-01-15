"""
Generator model API wrapper class.
"""

import torch
from torchvision.utils import save_image
from typing import List
import sys

# append parent directory
sys.path.append('..')

from modules import ConditionalGenerator


class GeneratorWrapper():
    def __init__(self, config):
        out_dir = config['output_directory']
        model_name = config['model_name']
        num_class = config['number_classes']
        input_dim = config['input_dimensions']
        z_dim = config['z_dimension']

        # model artifact file path
        model_file = '../{}{}_generator.pt'.format(out_dir, model_name)

        # initialize and load model
        self._generator = ConditionalGenerator(z_dim, num_class, input_dim[-1])
        self._generator.load_state_dict(torch.load(
            model_file, map_location=torch.device('cpu')))

        # initialize style vector distribution
        self._style_dist = torch.distributions.normal.Normal(
            torch.zeros(1, z_dim), torch.ones(1, z_dim))

        # sample style distribution to initialize style vector
        self._style_vec = self._style_dist.sample()

        # set number classes
        self._num_class = num_class

    def get_summary_dict(self):
        return {
            obj.strip(' ').split(':')[0].strip('()'):\
            obj.strip(' ').split(':')[1]\
            for obj in repr(self._generator).split('\n')[1:-1]}

    def sample_new_style(self):
        self._style_vec = self._style_dist.sample()

    def generate_image(self, labels: List[int], media_type='image/png'):
        # create 'multi-hot' distribution from labels list
        label_vec = torch.zeros(1, self._num_class)
        label_vec[0, labels] = 1./float(len(labels))
        
        # generate sample
        gen_out = self._generator(self._style_vec, label_vec)[0]

        print(gen_out)

        # save generated sample to image file
        if media_type == 'image/png':
            save_image(gen_out, 'gen_out.png')
        else:
            save_image(gen_out, 'gen_out.png')
