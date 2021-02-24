import torch
import torch.nn as nn
import numpy as np
import gan_flow_data

n_samples = 1876
samples_output_path = 'output/features_gan.csv'

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('Device: ' + device.type)

# Load saved model
netG = gan_flow_data.Generator(gan_flow_data.ngpu).to(device)
netG.load_state_dict(torch.load(gan_flow_data.generator_output_path))
netG.eval()
print('Loaded generator from \'' + gan_flow_data.generator_output_path + '\'')

# Generate samples
noise = torch.randn(n_samples, gan_flow_data.nz, device=device)
fake = netG(noise)
samples = fake.data.cpu().numpy()
# np.savetxt(samples_output_path, samples, delimiter=',', fmt='%f')
np.savetxt(samples_output_path, samples, delimiter=',', header=','.join(gan_flow_data.headers), comments='', fmt='%f')
print('Saved new samples to \'' + samples_output_path + '\'')
