"""
Launches a serving API for a pre-trained generator model.
"""

import yaml
import torch
import sys
from typing import Optional
from fastapi import FastAPI

# append parent directory
sys.path.append('..')

from modules import ConditionalGenerator


# parse configuration file
with open('../config.yaml', 'r') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

data_dir = config['dataset_directory']
out_dir = config['output_directory']
model_name = config['model_name']
num_class = config['number_classes']
input_dim = config['input_dimensions']
z_dim = config['z_dimension']

# model artifact file path
model_file = '../{}{}_generator.pt'.format(out_dir, model_name)

# initialize and load model
generator = ConditionalGenerator(z_dim, num_class, input_dim[-1])
generator.load_state_dict(torch.load(
    model_file, map_location=torch.device('cpu')))

app = FastAPI()

@app.get('/generator/summary/')
def post_model_architecture():
    return {
        obj.strip(' ').split(':')[0].strip('()'): obj.strip(' ').split(':')[1]
        for obj in repr(generator).split('\n')[1:-1]}
