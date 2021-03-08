import torch
import pandas as pd
import gan_flow_data


def generate(generator, headers, n_samples):
    noise = torch.randn(n_samples, gan_flow_data.nz)
    fake = generator(noise)
    samples = fake.data.cpu().numpy()
    return pd.DataFrame(samples, columns=headers)

