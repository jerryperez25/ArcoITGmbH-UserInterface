from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 0

# Filepath for input data
dataroot = 'output/features.csv'

# Filepath for generator model
generator_output_path = 'output/generator.pt'

# Number of workers for dataloader
workers = 2

# Number of training epochs
num_epochs = 3

# Batch size during training
batch_size = 8

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of features
nc = 16

# Number of one hot vectors for each categorical feature
n_one_hot = [12, 12, 3, 4, 3, 3]

# Size of feature maps in generator
nf = 47

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nf, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def OneHot(input):
    output = []
    for x in input:
        y_max = x[0]
        i = 0
        for j in range(len(x)):
            y = x[j]
            if y > y_max:
                y_max = y
                i = j
        x1 = [0.] * len(x)
        x1[i] = 1.
        output.append(x1)
    return torch.tensor(output)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.input_layer = nn.Linear(nz, 128, bias=False)
        self.hidden_layer = nn.Linear(128, 64, bias=False)
        self.output_layer = nn.Linear(64, nc, bias=False)
        self.actv_relu = nn.ReLU(True)
        self.actv_tanh = nn.Tanh()
        self.input_cat_layer0 = nn.Linear(1, n_one_hot[0], bias=False)
        self.input_cat_layer1 = nn.Linear(1, n_one_hot[1], bias=False)
        self.input_cat_layer2 = nn.Linear(1, n_one_hot[2], bias=False)
        self.input_cat_layer3 = nn.Linear(1, n_one_hot[3], bias=False)
        self.input_cat_layer4 = nn.Linear(1, n_one_hot[4], bias=False)
        self.input_cat_layer5 = nn.Linear(1, n_one_hot[5], bias=False)
        self.actv_softmax = nn.Softmax(dim=1)
        self.actv_onehot = OneHot

    def forward(self, input):
        # Sequential model
        x = self.input_layer(input)
        x = self.actv_relu(x)
        x = self.hidden_layer(x)
        x = self.actv_relu(x)
        x = self.output_layer(x)
        x = self.actv_tanh(x)

        # Categorical inputs
        x0 = x[:, 0] # Site A
        x0 = torch.reshape(x0, (x0.shape[0], 1))
        x1 = x[:, 1] # Site B
        x1 = torch.reshape(x1, (x1.shape[0], 1))
        x2 = x[:, 2] # Network Type A
        x2 = torch.reshape(x2, (x2.shape[0], 1))
        x3 = x[:, 3] # Network Type B
        x3 = torch.reshape(x3, (x3.shape[0], 1))
        x4 = x[:, 4] # Port A
        x4 = torch.reshape(x4, (x4.shape[0], 1))
        x5 = x[:, 5] # Port B
        x5 = torch.reshape(x5, (x5.shape[0], 1))
        x6 = x[:, 6:] # The rest

        # Parallel dense layers
        x0 = self.input_cat_layer0(x0)
        x0 = self.actv_softmax(x0)
        x0 = self.actv_onehot(x0)
        x1 = self.input_cat_layer1(x1)
        x1 = self.actv_softmax(x1)
        x1 = self.actv_onehot(x1)
        x2 = self.input_cat_layer2(x2)
        x2 = self.actv_softmax(x2)
        x2 = self.actv_onehot(x2)
        x3 = self.input_cat_layer3(x3)
        x3 = self.actv_softmax(x3)
        x3 = self.actv_onehot(x3)
        x4 = self.input_cat_layer4(x4)
        x4 = self.actv_softmax(x4)
        x4 = self.actv_onehot(x4)
        x5 = self.input_cat_layer5(x5)
        x5 = self.actv_softmax(x5)
        x5 = self.actv_onehot(x5)

        # Concat x's
        x = torch.cat((x0, x1, x2, x3, x4, x5, x6), 1)

        return x

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

if __name__ == '__main__':
    #manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Create the dataset
    dataset = TensorDataset(torch.Tensor(np.genfromtxt(dataroot, delimiter=',', skip_header=1)))

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    print('Training data loaded')

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print('Device: ' + device.type)

    # Create the Generator and the Discriminator
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, device=device)

    # Establish convention for real and fake labels during training
    real_label = 0.
    fake_label = 1.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print('Created generator and discriminator')

    # Training Loop
    print('Training started')
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, device=device)
            # Generate fake batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    print('Training finished')

    torch.save(netG.state_dict(), generator_output_path)
    print('Saved generator to \'' + generator_output_path + '\'')
