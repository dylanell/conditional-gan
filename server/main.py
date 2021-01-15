"""
Launches a serving API for a pre-trained generator model.
"""


import yaml
from model_wrapper import GeneratorWrapper
from typing import List
from fastapi import FastAPI
from fastapi.responses import FileResponse

# parse configuration file
with open('../config.yaml', 'r') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

# create generator wrapper with config params
generator_wrapper = GeneratorWrapper(config)

app = FastAPI()

@app.get('/api/summary')
async def get_summary_dict():
    return generator_wrapper.get_summary_dict()

@app.post('/api/generate-image')
async def generate_image(labels: List[int]):
    generator_wrapper.generate_image(labels, media_type='image/png')

    return FileResponse('gen_out.png', media_type='image/png')

@app.get('/api/sample-new-style', status_code=200)
async def sample_new_style():
    generator_wrapper.sample_new_style()

    return 'success'
