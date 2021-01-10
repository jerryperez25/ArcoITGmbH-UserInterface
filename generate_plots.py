import pandas as pd
from scipy import stats
# import matplotlib.pyplot as plt

# Import data
flow_data = pd.read_csv('sample_data/sample_data.csv', header=0)
ip_data = pd.read_csv('sample_data/ip_inventory.csv', header=0, index_col=0)

# Functions
def map_ips_to_col(address_name, col_name):
    site_list = []
    for ip in flow_data[address_name]:
        try:
            site_list.append(ip_data.at[ip, col_name])
        except:
            site_list.append('NaN')
    return site_list

def normalize_col(col):
    return stats.zscore(col)

def port_col_to_something(port_col):
    return port_col

# Feature engineering
features = pd.DataFrame(data={})

site_a_list = map_ips_to_col('Address A', 'Plant Area')
flow_data['Site A'] = site_a_list
features['Site A'] = site_a_list

site_b_list = map_ips_to_col('Address B', 'Plant Area')
flow_data['Site B'] = site_b_list
features['Site B'] = site_b_list

duration_z = normalize_col(flow_data['Duration'])
flow_data['Duration (z)'] = duration_z
features['Duration (z)'] = duration_z

print(features)
