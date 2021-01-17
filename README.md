# conditional-gan

PyTorch implementation of Conditional [Improved Wasserstein Generative Adversarial Network (GAN)](https://arxiv.org/pdf/1704.00028.pdf) on the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/). The conditional GAN introduces a one-hot vector that is also provided as input to the generator of the GAN along with the original z input sample from the multivariate normal distribution. This additional one-hot vector takes on the role of encoding the "class" for the sample to be generated while the z vector controls the "style". Therefore, by individually altering the z vector or the one-hot vector provided to the generator, one can control the "style" and the "label" for generated samples, respectively.

| ![](conditional_gan/artifacts/gen.gif) |
| :-: |
| *Generator outputs with constant inputs during training. Column labels can be found in [`gen_gif_cols.txt`](https://github.com/dylanell/conditional-gan/blob/main/artifacts/readme_gen_gif_cols.txt)* |

### Project Structure:

```
conditional-gan/
├── conditional_gan
│   ├── artifacts
│   │   ├── classifier.pt
│   │   ├── critic.pt
│   │   ├── generator.pt
│   │   ├── gen.gif
│   │   └── gif_cols.txt
│   ├── config.yaml
│   ├── data
│   │   ├── datasets.py
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── __init__.py
│   ├── modules.py
│   ├── train.py
│   └── util
│       ├── distributions.py
│       └── __init__.py
├── docker-compose.yaml
├── Dockerfile
├── README.md
├── requirements.txt
└── server
    ├── api.py
    ├── gen_out.png
    ├── __init__.py
    └── wrappers.py
```

### Runtime:

- Python 3.8.5

### Install Requirements:

```
$ pip install -r requirements.txt
```

### Image Dataset Format:

This project assumes you have the MNIST dataset pre-configured locally on your machine in the format described below. My [dataset-helpers](https://github.com/dylanell/dataset-helpers) Github project also contains tools that perform this local configuration automatically within the `mnist` directory of the project.

The MNIST dataset consists of images of written numbers (0-9) with corresponding labels. The dataset can be accessed a number of ways using Python packages (`mnist`, `torchvision`, `tensorflow_datasets`, etc.), or it can be downloaded directly from the [MNIST homepage](http://yann.lecun.com/exdb/mnist/). In order to develop image-based data pipelines in a standard way, we organize the MNIST dataset into training/testing directories of raw image files (`png` or `jpg`) accompanied by a `csv` file listing one-to-one correspondences between the image file names and their label. This "generic image dataset format" is summarized by the directory tree structure below.

```
mnist_png/
├── test
│   ├── test_image_01.png
│   ├── test_image_02.png
│   └── ...
├── test_labels.csv
├── train
│   ├── train_image_01.png
│   ├── train_image_02.png
│   └── ...
└── train_labels.csv
```

Each labels `csv` file has the format:

```
Filename, Label
train_image_01.png, 4
train_image_02.png, 7
...
```

If you would like to re-use the code here to work with other image datasets, just format any new image dataset to follow the outline above and be sure to edit corresponding hyperparameters in the `config.yaml` file.

### Training:

Model and training files for the Conditional GAN are located within the `conditional_gan` subdirectory. Configuration and training parameters can be controlled by editing the `config.yaml` within this directory.

To train the model, navigate to the `conditional_gan` directory and run:

```
$ python train.py
```

The training script will generate model artifacts to the `artifacts/` directory.

### Serving with Python:

This project uses [FastAPI](https://fastapi.tiangolo.com/) to setup a model serving API for a pre-trained generator model. The following command will spin up the API application on `http://0.0.0.0:8080`.

```
$ python api.py
```

Swagger UI interactive API documentation can be viewed at the `/docs` endpoint.

### Serving with Docker:

First, make sure you have [Docker](https://www.docker.com/) installed along with the [docker-compose](https://docs.docker.com/compose/install/) cli. Run the following command from the top level of this project directory to build the required Docker image (on the first run only), and spin up an isolated container that runs the API application on `http://0.0.0.0:8080`.

```
$ docker-compose up
```

### References:

1. [Improved Wasserstein GAN](https://arxiv.org/pdf/1704.00028.pdf)
2. [Creative Adversarial Network (CAN)](https://arxiv.org/pdf/1706.07068.pdf)
