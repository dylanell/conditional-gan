# conditional-gan

PyTorch implementation of Conditional [Improved Wasserstein Generative Adversarial Network (GAN)](https://arxiv.org/pdf/1704.00028.pdf) on the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/). The conditional GAN introduces a one-hot vector that is also provided as input to the generator of the GAN along with the original z input sample from the multivariate normal distribution. This additional one-hot vector takes on the role of encoding the "class" for the sample to be generated while the z vector controls the "style". Therefore, by individually altering the z vector or the one-hot vector provided to the generator, one can control the "style" and the "label" for generated samples, respectively.

| ![](conditional_gan/artifacts/readme_gen.gif) |
| :-: |
| *Generator outputs with constant inputs during training. Column labels can be found in [`readme_gen_gif_cols.txt`](https://github.com/dylanell/conditional-gan/blob/main/artifacts/readme_gen_gif_cols.txt)* |

### Project Structure:

```
conditional-gan/
├── conditional_gan
│   ├── artifacts
│   │   ├── pan_02_classifier.pt
│   │   ├── pan_02_critic.pt
│   │   ├── pan_02_generator.pt
│   │   ├── pan_02_gif_cols.txt
│   │   ├── readme_gen.gif
│   │   └── readme_gen_gif_cols.txt
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
├── README.md
├── requirements.txt
└── server
    ├── gen_out.png
    ├── __init__.py
    ├── main.py
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

Model and training files for the Conditional GAN are located within the `conditional_gan` model directory. Configuration and training parameters can be controlled by editing the `config.yaml` within the model directory.

To train the model, navigate to the `conditional_gan` directory and run:

```
$ python train.py
```

The training script will generate several files to an output directory (local `artifacts/` directory by default) including example images and model parameter files.

### Serving:

This project uses [FastAPI](https://fastapi.tiangolo.com/) to setup a model serving API for a pre-trained generator model.

To spin up the server application on port 8080, navigate to the `server` directory and run:

```
$ python main.py -p 8080
```

Swagger UI interactive API documentation can be viewed at the `/docs` endpoint.

When using the `/api/generate-image` endpoint, the server application uses a `*_generator.pt` model artifact (written to the `artifacts/` directory after running the training script) to load a pre-trained generator model, computes an output given a style and label vector, writes this output locally as an image file, and returns this image file as a [`fastapi.response.FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse) back to the client.  

### References:

1. [Improved Wasserstein GAN](https://arxiv.org/pdf/1704.00028.pdf)
2. [Creative Adversarial Network (CAN)](https://arxiv.org/pdf/1706.07068.pdf)
