import torch
import pandas as pd
import gan_flow_data

samples_output_path = 'output/features_gan.csv'


def generate(generator, headers, n_samples):
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 0

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print('Device: ' + device.type)

    noise = torch.randn(n_samples, gan_flow_data.nz)
    fake = generator(noise)
    samples = fake.data.cpu().numpy()
    return pd.DataFrame(samples, columns=headers)

