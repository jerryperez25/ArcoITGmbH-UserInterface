import torch
import torch.nn as nn
import numpy as np

# Filepath for generator model
generator_output_path = 'output/generator.pt'

# Filepath for new samples
samples_output_path = 'output/features_gan.csv'

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
nf = 47

n_samples = 10

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(True),
            nn.Linear(64, nf, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

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
np.savetxt(samples_output_path, samples, delimiter=',')
print('Saved new samples to \'' + samples_output_path + '\'')
