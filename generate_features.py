import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import sys

# Import data
flow_data = None
ip_data = None
if len(sys.argv) > 4:
    flow_data1 = pd.read_csv(sys.argv[1], header=0)
    flow_data2 = pd.read_csv(sys.argv[2], header=0)
    flow_data = pd.concat([flow_data1, flow_data2])
    flow_data.to_csv('output/flow_test.csv', index=False)
    ip_data = pd.read_csv(sys.argv[3], header=0, index_col=0)
else:
    flow_data = pd.read_csv(sys.argv[1], header=0)
    ip_data = pd.read_csv(sys.argv[2], header=0, index_col=0)

# Functions
def map_ips_to_col(address_name, col_name):
    site_list = []
    for ip in flow_data[address_name]:
        try:
            name = ip_data.at[ip, col_name]
            if type(name) is not np.ndarray:
                site_list.append(name)
            else:
                site_list.append(name[0])
        except:
            site_list.append('NaN')
    return site_list

def normalize_col(col):
    return stats.zscore(col)

def clean_int_data(data):
    y = []
    for x in data:
        if type(x) is str:
            y.append(int(x.replace(',', '')))
        else:
            y.append(x)
    return y

def port_col_to_class(port_col):
    common_ports = {502: 'modbus', 80: 'http', 443: 'http', 5450: 'tiepie', 8081: 'couchbase', 14000: 'tcp', 12350: 'tcp', 4353: 'tcp', 3840: 'tcp', 139: 'microsoft ds', 445: 'microsoft ds', 194: 'irc', 6667: 'irc'}
    return list(map(lambda x: common_ports[x] if x in common_ports else 'NaN', port_col))

def encode_one_hot(feature_name, feature):
    feature_shaped = np.reshape(feature, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    one_hot = enc.fit_transform(feature_shaped).toarray()
    one_hot_names = enc.get_feature_names([feature_name])
    return one_hot_names, one_hot

def add_one_hot_features(features, one_hot_names, one_hot):
    one_hot_T = np.transpose(one_hot)
    for i in range(len(one_hot_names)):
        features[one_hot_names[i]] = one_hot_T[i]
    return features

# Feature engineering
features = pd.DataFrame(data={})

site_a_features = encode_one_hot('Site A', map_ips_to_col('Address A', 'Plant Area'))
features = add_one_hot_features(features, site_a_features[0], site_a_features[1])

site_b_features = encode_one_hot('Site B', map_ips_to_col('Address B', 'Plant Area'))
features = add_one_hot_features(features, site_b_features[0], site_b_features[1])

network_type_a_features = encode_one_hot('Network Type A', map_ips_to_col('Address A', 'Network Type'))
features = add_one_hot_features(features, network_type_a_features[0], network_type_a_features[1])

network_type_b_features = encode_one_hot('Network Type B', map_ips_to_col('Address B', 'Network Type'))
features = add_one_hot_features(features, network_type_b_features[0], network_type_b_features[1])

port_a_features = encode_one_hot('Port A', port_col_to_class(flow_data['Port A']))
features = add_one_hot_features(features, port_a_features[0], port_a_features[1])

port_b_features = encode_one_hot('Port B', port_col_to_class(flow_data['Port B']))
features = add_one_hot_features(features, port_b_features[0], port_b_features[1])

flow_data['Packets'] = clean_int_data(flow_data['Packets'])
packets_z = normalize_col(flow_data['Packets'])
features['Packets (z)'] = packets_z

flow_data['Bytes'] = clean_int_data(flow_data['Bytes'])
bytes_z = normalize_col(flow_data['Bytes'])
features['Bytes (z)'] = bytes_z

flow_data['Packets A -> B'] = clean_int_data(flow_data['Packets A -> B'])
packets_ab_z = normalize_col(flow_data['Packets A -> B'])
features['Packets A -> B (z)'] = packets_ab_z

flow_data['Bytes A -> B'] = clean_int_data(flow_data['Bytes A -> B'])
bytes_ab_z = normalize_col(flow_data['Bytes A -> B'])
features['Bytes A -> B (z)'] = bytes_ab_z

flow_data['Packets B -> A'] = clean_int_data(flow_data['Packets B -> A'])
packets_ba_z = normalize_col(flow_data['Packets B -> A'])
features['Packets B -> A (z)'] = packets_ba_z

flow_data['Bytes B -> A'] = clean_int_data(flow_data['Bytes B -> A'])
bytes_ba_z = normalize_col(flow_data['Bytes B -> A'])
features['Bytes B -> A (z)'] = bytes_ba_z

rel_start_z = normalize_col(flow_data['Rel Start'])
features['Rel Start (z)'] = rel_start_z

duration_z = normalize_col(flow_data['Duration'])
features['Duration (z)'] = duration_z

bit_rate_ab_z = normalize_col(flow_data['Bits/s A -> B'])
features['Bits/s A -> B (z)'] = bit_rate_ab_z

bit_rate_ba_z = normalize_col(flow_data['Bits/s B -> A'])
features['Bits/s B -> A (z)'] = bit_rate_ba_z

# Save to file
features.to_csv(sys.argv[len(sys.argv) - 1], index=False)
