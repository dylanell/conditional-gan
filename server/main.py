"""
Launches a serving API for a pre-trained generator model.
"""


import yaml
import uvicorn
from model_wrapper import GeneratorWrapper
from typing import List
from fastapi import FastAPI,Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# parse configuration file
with open('../config.yaml', 'r') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

# create generator wrapper with config params
generator_wrapper = GeneratorWrapper(config)

app = FastAPI()

# Serve the React App Frontend
@app.get('/')
def index():
    return FileResponse('../client/build/index.html', media_type='text/html')

app.mount('/static', StaticFiles(directory='../client/build/static'), name='static')

@app.get('/api/summary')
def get_summary_dict():
    return generator_wrapper.get_summary_dict()

@app.post('/api/generate-image')
def generate_image(labels: List[int]):
    generator_wrapper.generate_image(labels, media_type='image/png')
    return FileResponse('gen_out.png', media_type='image/png')

@app.get('/api/sample-new-style', status_code=200)
def sample_new_style():
    generator_wrapper.sample_new_style()
    return 'success'

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=8080, reload=True)