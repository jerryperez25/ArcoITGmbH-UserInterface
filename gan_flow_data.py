from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

debug_level = 0


def print_debug(print_out):
    if debug_level > 0:
        print(print_out)

# Set random seed for reproducibility
# manualSeed = 0

# Number of workers for dataloader
workers = 2

# Number of training epochs
num_epochs = 3

# Batch size during training
batch_size = 8

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Learning rate for optimizers
lrD = 0.00002
lrG = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0


def generate(features):
    # Number of features
    nc = 0
    # Number of one hot vectors for each categorical feature
    n_one_hot = []
    # Size of feature maps in generator
    nf = 0

    # Create the dataset
    data_tensor = torch.from_numpy(features.values)
    dataset = TensorDataset(data_tensor.float())
    headers = features.columns

    prev_one_hot = ''
    for h in headers:
        if '_' in h:
            one_hot = h.split('_')[0]
            if one_hot == prev_one_hot:
                n_one_hot[-1] += 1
            else:
                prev_one_hot = one_hot
                nc += 1
                n_one_hot.append(1)
        nf += 1
    # TCP/UDP
    nc += 1
    n_one_hot.append(2)

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

    def Binary(input):
        output = []
        for x in input:
            output_x = []
            for y in x:
                output_x.append(1. if y > 0.5 else 0.)
            output.append(output_x)
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
            self.actv_softmax = nn.Softmax(dim=1)
            self.actv_onehot = OneHot
            self.actv_binary = Binary
            self.cat_layers = []
            for n in range(len(n_one_hot)):
                self.cat_layers.append([
                    nn.Linear(1, 16, bias=False),
                    nn.Linear(16, n_one_hot[n], bias=False)
                ])

        def forward(self, input):
            # Sequential model
            x = self.input_layer(input)
            x = self.actv_relu(x)
            x = self.hidden_layer(x)
            x = self.actv_relu(x)
            x = self.output_layer(x)
            x = self.actv_tanh(x)

            x1 = None
            for n in range(len(n_one_hot)):
                x0 = x[:, n]
                x0 = torch.reshape(x0, (x0.shape[0], 1))

                x0 = self.cat_layers[n][0](x0)
                x0 = self.cat_layers[n][1](x0)
                # x0 = self.cat_layers[n][2](x0)
                x0 = self.actv_softmax(x0)
                x0 = self.actv_onehot(x0)

                if x1 is None:
                    x1 = x0
                else:
                    x1 = torch.cat((x1, x0), 1)

            return x1

    # custom weights initialization called on netG and netD
    # try orthogonal matrix initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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
    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

    # Training Loop
    print_debug('Training started')
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
                output_stats = '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch + 1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
                print_debug(output_stats)
    print_debug('Training finished')

    return netG, headers
