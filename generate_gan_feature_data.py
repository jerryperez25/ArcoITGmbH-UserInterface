import torch
import pandas as pd
import gan_flow_data


def generate(generator, headers, n_samples):
    noise = torch.randn(int(n_samples), gan_flow_data.nz)
    fake = generator(noise)
    samples = fake.data.cpu().numpy()
    return pd.DataFrame(samples, columns=headers)


if __name__ == "__main__":
    features_0 = pd.read_csv('data/features.csv')
    generator, headers = gan_flow_data.generate(features_0)
    features_1 = generate(generator, headers, features_0.shape[0] / 2)
    features_1.to_csv('data/features_gan.csv', index=False)
