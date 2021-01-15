"""
Launches a serving API for a pre-trained generator model.
"""


import yaml
import torch
from torchvision.utils import save_image
import sys
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

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

# wrapper for generator style and label inputs
class GeneratorInput(BaseModel):
    style_vector: List[float]
    label_vector: List[float]

app = FastAPI()

@app.get('/api/summary/')
async def summary_enpoint():
    return {
        obj.strip(' ').split(':')[0].strip('()'): obj.strip(' ').split(':')[1]
        for obj in repr(generator).split('\n')[1:-1]}

@app.post('/api/generate/')
async def generate_endpoint(input: GeneratorInput):
    # validate input data

    # convert inputs to torch tensors
    z_vec = torch.tensor(input.style_vector).unsqueeze(0)
    y_vec = torch.tensor(input.label_vector).unsqueeze(0)

    # generate sample
    gen_out = generator(z_vec, y_vec)[0]

    # save gen_out image
    save_image(gen_out, 'gen_out.png')

    return FileResponse('gen_out.png', media_type='image/png')
