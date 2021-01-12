"""
Train a Conditional Wasserstein GAN.
"""

import torch
import yaml
import time
from torchvision.utils import make_grid, save_image

from util.data_pipeline import process_image_dataset
from modules import Critic, ConditionalGenerator, Classifier

def main():
    # parse config
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    data_dir = config['dataset_directory']
    out_dir = config['output_directory']
    model_name = config['model_name']
    num_class = config['number_classes']
    input_dim = config['input_dimensions']
    batch_size = config['batch_size']
    num_epoch = config['number_epochs']
    learn_rate = config['learning_rate']
    z_dim = config['z_dimension']
    num_work = config['number_workers']

    # try to get gpu device, if not just use cpu
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() and config['acceleration']
        else 'cpu')
    print('[INFO]: using \'{}\' device'.format(device))

    # create image datasets/dataloaders
    dataset_dct = process_image_dataset(
        data_dir, image_size=input_dim[:-1],
        batch_size=batch_size, num_workers=num_work)

    # initialize critic (CNN) model
    critic = Critic(input_dim[-1], 1)

    # initialize generator (TransposeCNN) model
    generator = ConditionalGenerator(z_dim, num_class, input_dim[-1])

    # initialize classifier (CNN) model
    classifier = Classifier(input_dim[-1], num_class)

    # put model son training device
    critic.to(device)
    generator.to(device)
    classifier.to(device)

    # initialize normal distribution to sample generator style
    z_dist = torch.distributions.normal.Normal(
        torch.zeros(batch_size, z_dim),
        torch.ones(batch_size, z_dim))

    # initialize one-hot categorical distribution to sample generator class
    y_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
        (1./float(num_class)) * torch.ones(batch_size, num_class))

    # initialize uniform distribution to sample eps vals for img interpolations
    eps_dist = torch.distributions.uniform.Uniform(
        torch.zeros(batch_size, 1, 1, 1),
        torch.ones(batch_size, 1, 1, 1))

    # initialize optimizers
    critic_opt = torch.optim.Adam(critic.parameters(), lr=learn_rate)
    generator_opt = torch.optim.Adam(generator.parameters(), lr=learn_rate)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=learn_rate)

    # cross entropy loss for classifier
    classifier_loss_fn = torch.nn.CrossEntropyLoss()

    # run through epochs
    for e in range(num_epoch):
        # get epoch start time
        epoch_start = time.time()

        # initialize epoch accumulators
        wass_dist_acc = 0
        classifier_loss_acc = 0

        # run through batches
        for i, batch in enumerate(dataset_dct['train_ldr']):
            # parse batch
            real_x = batch['image'].to(device)
            real_y = batch['label'].to(device)

            # get number of samples in batch
            bs = real_x.shape[0]

            # sample from input distributions and clip based on batch size
            z_sample = z_dist.sample()[:bs].to(device)
            y_sample = y_dist.sample()[:bs].to(device)
            eps_sample = eps_dist.sample()[:bs].to(device)

            # generate a batch of fake images
            fake_x = generator(z_sample, y_sample)

            # interpolate between real x and fake x
            int_x = (eps_sample * real_x) + ((1. - eps_sample) * fake_x)

            # compute critic outputs from real, fake, and interpolated images
            real_c = critic(real_x)
            fake_c = critic(fake_x)
            int_c = critic(int_x)

            # compute classifier outputs from real and fake images
            real_logits = classifier(real_x)
            fake_logits = classifier(fake_x)

            # classifier loss for fake and real images
            real_classifier_loss = classifier_loss_fn(real_logits, real_y)
            fake_classifier_loss = classifier_loss_fn(
                fake_logits, torch.argmax(y_sample, dim=1))

            # compute gradient of critic output w.r.t interpolated images
            critic_grad = torch.autograd.grad(
                outputs=int_c,
                inputs=int_x,
                grad_outputs=torch.ones_like(int_c),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            # compute wasserstein distance
            wass_dist = torch.mean(real_c - fake_c)

            # compute mean of normed critic gradients
            critic_grad_mean_norm = torch.mean(torch.norm(
                critic_grad, p=2, dim=(2, 3)))

            # lagrangian multiplier for critic gradient penalty
            # (push critic_grad_mean_norm -> 1)
            critic_grad_penalty = (critic_grad_mean_norm - 1.) ** 2

            # generator loss (minimize wasserstein distance and fake
            # classifier loss)
            generator_loss = wass_dist + fake_classifier_loss

            # critic loss (maximize wasserstein distance with gradient penalty)
            critic_loss = (10.0 * critic_grad_penalty) - wass_dist

            # NOTE: Currently must update critic and generator separately.
            # If both are updated within the same loop, either updating
            # doesn't happen, or an inplace operator error occurs which
            # prevents gradient computation, depending on the ordering of
            # the zero_grad(), backward(), step() calls. ???
            if i % 10 == 9:
                # update just the generator (every 10th step)
                generator_opt.zero_grad()
                generator_loss.backward()
                generator_opt.step()

            else:
                # update classifier
                classifier_opt.zero_grad()
                real_classifier_loss.backward()
                classifier_opt.step()

                # update just the critic and classifier
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

            # update epoch metric accumulators
            wass_dist_acc += wass_dist.item()
            classifier_loss_acc += real_classifier_loss.item()

        # end epoch

        # compute epoch time
        epoch_time = time.time() - epoch_start

        # compute epoch metrics
        wass_dist = wass_dist_acc / (i + 1)
        classifier_loss = classifier_loss_acc / (i + 1)

        # print epoch metrics
        template = '[INFO]: Epoch {}, Epoch Time {:.2f}s, ' \
                   'Wasserstein Distance: {:.4f}, Classifier Loss: {:.4f}'
        print(template.format(e + 1, epoch_time, wass_dist, classifier_loss))

        # number of different styles to generate
        num_styles = 8

        # sample num_styles from z and repeat interleave rows for num_class
        z_sample = z_dist.sample()[:num_styles].to(device)
        z_stack = torch.repeat_interleave(z_sample, num_class, dim=0)

        # create num_styles copies of num_class identity array
        y_stack = torch.eye(num_class).repeat(num_styles, 1).to(device)

        # generate 10 styles of 1-num_class
        x_stack = generator(z_stack, y_stack)

        # reshape into 10xnum_class image
        b, c, h, w = x_stack.shape
        x_grid = make_grid(x_stack, nrow=num_class)

        # save image
        save_image(x_grid, '{}epoch_{}.png'.format(out_dir, e+1))

if __name__ == '__main__':
    main()
