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
from fastapi.middleware.cors import CORSMiddleware

# parse configuration file
with open('../config.yaml', 'r') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

# create generator wrapper with config params
generator_wrapper = GeneratorWrapper(config)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the React App Frontend
@app.get('/')
def index():
    return FileResponse('../client/build/index.html', media_type='text/html')

app.mount('/static', StaticFiles(directory='../client/build/static'), name='static')

@app.get('/api/summary')
async def get_summary_dict():
    return generator_wrapper.get_summary_dict()

@app.post('/api/generate-image')
async def generate_image(labels: List[float]):
    generator_wrapper.generate_image(labels, media_type='image/png')
    return FileResponse('gen_out.png', media_type='image/png')

@app.get('/api/sample-new-style', status_code=200)
async def sample_new_style():
    generator_wrapper.sample_new_style()
    return 'success'

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=8080, reload=True)