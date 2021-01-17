"""
Model serving API with FastAPI.
"""


import os
import yaml
from typing import List
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

from wrappers import GeneratorWrapper


def main():
    # file paths
    config_file = '../conditional_gan/config.yaml'
    generator_file = '../conditional_gan/artifacts/generator.pt'

    # check config -> artifacts consistency
    if os.path.getmtime(config_file) > os.path.getmtime(generator_file):
        print('[WARNING]: Model artifacts precede last edit of config file. '\
            'Config params may be inconsistent with model artifacts.')

    # parse configuration file
    with open(config_file, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # create generator wrapper with config params
    generator_wrapper = GeneratorWrapper(config)

    app = FastAPI()

    @app.get('/api/summary')
    async def get_summary_dict():
        return generator_wrapper.get_summary_dict()

    @app.post('/api/generate-image')
    async def generate_image(labels: List[float]):
        generator_wrapper.generate_image(labels, media_type='image/png')
        return FileResponse('gen_out.png', media_type='image/png')

    @app.get('/api/sample-new-style')
    async def sample_new_style():
        generator_wrapper.sample_new_style()
        return 'success'

    # run app
    uvicorn.run(app, host='0.0.0.0', port=8080, reload=False)


if __name__ == '__main__':
    main()
