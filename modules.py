"""
Pytorch nn modules.
"""

import torch


class Critic(torch.nn.Module):
    # initialize the base class and define network layers
    def __init__(self, in_chan, out_dim):
        # run base initializer
        super(Critic, self).__init__()

        # activation functions
        self.hid_act = torch.nn.ReLU()
        self.out_act = torch.nn.Identity()

        # convolutional layers
        self.conv_1 = torch.nn.Conv2d(in_chan, 32, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv_4 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv_5 = torch.nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # norm layers
        self.norm_1 = torch.nn.LayerNorm([32, 32, 32])
        self.norm_2 = torch.nn.LayerNorm([64, 16, 16])
        self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
        self.norm_4 = torch.nn.LayerNorm([256, 4, 4])
        self.norm_5 = torch.nn.LayerNorm([512, 2, 2])

        # fully connected layers
        self.fc_1 = torch.nn.Linear(512 * 2 * 2, out_dim)

    # forward propagate input x through the network
    def forward(self, x):
        z = self.hid_act(self.conv_1(x))
        z = self.hid_act(self.conv_2(z))
        z = self.hid_act(self.conv_3(z))
        z = self.hid_act(self.conv_4(z))
        z = self.hid_act(self.conv_5(z))
        z = torch.flatten(z, start_dim=1)
        z = self.out_act(self.fc_1(z))

        return z


class Classifier(torch.nn.Module):
    # initialize the base class and define network layers
    def __init__(self, in_chan, out_dim):
        # run base initializer
        super(Classifier, self).__init__()

        # activation functions
        self.hid_act = torch.nn.ReLU()
        self.out_act = torch.nn.Identity()

        # convolutional layers
        self.conv_1 = torch.nn.Conv2d(in_chan, 32, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv_4 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv_5 = torch.nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # norm layers
        self.norm_1 = torch.nn.LayerNorm([32, 32, 32])
        self.norm_2 = torch.nn.LayerNorm([64, 16, 16])
        self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
        self.norm_4 = torch.nn.LayerNorm([256, 4, 4])
        self.norm_5 = torch.nn.LayerNorm([512, 2, 2])

        # fully connected layers
        self.fc_1 = torch.nn.Linear(512 * 2 * 2, out_dim)

    # forward propagate input x through the network
    def forward(self, x):
        z = self.hid_act(self.conv_1(x))
        z = self.hid_act(self.conv_2(z))
        z = self.hid_act(self.conv_3(z))
        z = self.hid_act(self.conv_4(z))
        z = self.hid_act(self.conv_5(z))
        z = torch.flatten(z, start_dim=1)
        z = self.out_act(self.fc_1(z))

        return z


class ConditionalGenerator(torch.nn.Module):
    # initialize the base class and define network layers
    def __init__(self, in_dim, num_class, out_chan):
        # run base initializer
        super(ConditionalGenerator, self).__init__()

        # activation functions
        self.hid_act = torch.nn.ReLU()
        self.out_act = torch.nn.Tanh()

        # fully connected layers
        self.fc_1 = torch.nn.Linear(num_class, in_dim)
        self.fc_2 = torch.nn.Linear(in_dim, 512 * 2 * 2)

        # convolutional layers
        self.conv_1 = torch.nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.conv_2 = torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.conv_4 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.conv_5 = torch.nn.ConvTranspose2d(
            32, out_chan, 3, stride=1, padding=1)

        # norm layers
        self.norm_1 = torch.nn.LayerNorm([512, 2, 2])
        self.norm_2 = torch.nn.LayerNorm([256, 4, 4])
        self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
        self.norm_4 = torch.nn.LayerNorm([64, 16, 16])
        self.norm_5 = torch.nn.LayerNorm([32, 32, 32])

    # forward propagate inputs z and y throgh the network
    def forward(self, z, y):
        bs = z.shape[0]
        x = z * self.fc_1(y)
        x = self.hid_act(self.fc_2(x))
        x = torch.reshape(x, [bs, 512, 2, 2])
        x = self.norm_1(x)
        x = self.hid_act(self.conv_1(x, output_size=(bs, 256, 4, 4)))
        x = self.norm_2(x)
        x = self.hid_act(self.conv_2(x, output_size=(bs, 128, 8, 8)))
        x = self.norm_3(x)
        x = self.hid_act(self.conv_3(x, output_size=(bs, 64, 16, 16)))
        x = self.norm_4(x)
        x = self.hid_act(self.conv_4(x, output_size=(bs, 32, 32, 32)))
        x = self.norm_5(x)
        x = self.out_act(self.conv_5(x, output_size=(bs, 1, 32, 32)))

        return x
