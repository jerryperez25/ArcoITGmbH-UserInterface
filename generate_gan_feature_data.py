import torch
import torch.nn as nn
import numpy as np

# Filepath for generator model
generator_output_path = 'output/generator.pt'

# Filepath for new samples
samples_output_path = 'output/features_gan.csv'

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of features
nc = 16

# Number of one hot vectors for each categorical feature
n_one_hot = [12, 12, 3, 4, 3, 3]

# Size of feature maps in generator
nf = 47

n_samples = 10

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

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
        x1 = [0] * len(x)
        x1[i] = 1
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

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('Device: ' + device.type)

# Load saved model
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load(generator_output_path))
netG.eval()
print('Loaded generator from \'' + generator_output_path + '\'')

# Generate samples
noise = torch.randn(n_samples, nz, device=device)
fake = netG(noise)
samples = fake.data.cpu().numpy()
np.savetxt(samples_output_path, samples, delimiter=',', fmt='%f')
print('Saved new samples to \'' + samples_output_path + '\'')
