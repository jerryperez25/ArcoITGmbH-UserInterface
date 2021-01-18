import pandas as pd
import numpy as np
import math
from faker import Faker
import os
import sys

zero_min = 0.001
bytes_max = 25000000
packet_size_max = 55000
rel_start_max = 1.0
bit_rate_max = 3000000

fake = Faker()
Faker.seed(0)

ip_data = pd.read_csv('sample_data/ip_inventory.csv', header=0)
ip_list = ip_data['LAN IP Address']

data = pd.DataFrame(columns=['Address A', 'Port A', 'Address B', 'Port B', 'Packets', 'Bytes', 'Packets A -> B', 'Bytes A -> B', 'Packets B -> A', 'Bytes B -> A', 'Rel Start', 'Duration', 'Bits/s A -> B', 'Bits/s B -> A'])

rel_start = 0.0
for i in range(int(sys.argv[1])):
    bytes_ab = np.random.randint(0, bytes_max)
    bytes_ba = np.random.randint(0, bytes_max)
    packet_size = np.random.randint(1, packet_size_max)
    packets_ab = math.ceil(bytes_ab / packet_size)
    packets_ba = math.ceil(bytes_ba / packet_size)
    bit_rate_ab = np.random.uniform(zero_min, bit_rate_max)
    bit_rate_ba = np.random.uniform(zero_min, bit_rate_max)
    data.loc[i] = [
        np.random.choice(ip_list),
        fake.port_number(),
        np.random.choice(ip_list),
        fake.port_number(),
        packets_ab + packets_ba,
        bytes_ab + bytes_ba,
        packets_ab,
        bytes_ab,
        packets_ba,
        bytes_ba,
        rel_start,
        bytes_ab / bit_rate_ab + bytes_ba / bit_rate_ba,
        bit_rate_ab,
        bit_rate_ba
    ]
    rel_start += np.random.uniform(zero_min, rel_start_max)

# Save to file
if not os.path.exists('output'):
    os.makedirs('output')
data.to_csv('output/random_flow_data.csv', index=False)
