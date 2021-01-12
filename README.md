# conditional-wgan

PyTorch implementation of Conditional [Improved Wasserstein Generative Adversarial Network (GAN)](https://arxiv.org/pdf/1704.00028.pdf) on the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/). The conditional GAN introduces a one-hot vector that is also provided as input to the generator of the GAN along with the original z input sample from the multivariate normal distribution. This additional one-hot vector takes on the role of encoding the "class" for the sample to be generated while the z vector controls the "style". Therefore, by individually altering the z vector or the one-hot vector provided to the generator, one can control the "style" and the "label" for generated samples, respectively.

| ![](artifacts/gen.gif) |
| :-: |
| *Some generator outputs during training with each column corresponding to a conditioned digit class in [0-9]* |

### Environment:

- Python 3.8.5

### Install Python Packages:

```
$ pip install -r requirements.txt
```

### Image Dataset Format:

This project assumes you have the MNIST dataset pre-configured locally on your machine in the format described below. My [dataset-helpers](https://github.com/dylanell/dataset-helpers) Github project also contains tools that perform this local configuration automatically within the `mnist` directory of the project.

The MNIST dataset consists of images of written numbers (0-9) with corresponding labels. The dataset can be accessed a number of ways using Python packages (`mnist`, `torchvision`, `tensorflow_datasets`, etc.), or it can be downloaded directly from the [MNIST homepage](http://yann.lecun.com/exdb/mnist/). In order to develop image-based data pipelines in a standard way, we organize the MNIST dataset into training/testing directories of raw image files (`png` or `jpg`) accompanied by a `csv` file listing one-to-one correspondences between the image file names and their label. This "generic image dataset format" is summarized by the directory tree structure below.

```
dataset_directory/
|__ train_labels.csv
|__ test_labels.csv
|__ train/
|   |__ train_image_01.png
|   |__ train_image_02.png
|   |__ ...
|__ test/
|   |__ test_image_01.png
|   |__ test_image_02.png
|   |__ ...   
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

Training hyperparameters are pulled from the `config.yaml` file and can be changed by editing the file contents.

Train the model by running:

```
$ python train.py
```

The training script will save images of generated samples to the output directory and use many of these images to produce a GIF after which they will be deleted.

### References:

1. Wasserstein GAN:

 - https://arxiv.org/pdf/1704.00028.pdf
