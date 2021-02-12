import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sys

NaN = 'NaN'
common_ports_dict = {502: 'modbus', 80: 'http', 443: 'http', 5450: 'tiepie', 8081: 'couchbase', 14000: 'tcp', 12350: 'tcp', 4353: 'tcp', 3840: 'tcp', 139: 'microsoft ds', 445: 'microsoft ds', 194: 'irc', 6667: 'irc'}
common_port_names = []
for p in common_ports_dict.values():
    if not p in common_port_names:
        common_port_names.append(p)
common_port_names.append(NaN)

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
            if type(name) is not pd.core.series.Series:
                site_list.append(name)
            else:
                site_list.append(name[0])
        except:
            site_list.append(NaN)
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
    return list(map(lambda x: common_ports_dict[x] if x in common_ports_dict else NaN, port_col))

def getCategories(column):
    d = ip_data[column]
    categories = []
    for i in range(len(d)):
        x = d[i]
        if not x in categories:
            categories.append(x)
    categories.append(NaN)
    return categories

def encode_one_hot_2(feature_name_a, feature_name_b, categories, features_a, features_b):
    categories.sort()
    enc = OneHotEncoder(categories=(categories,))
    features_a = np.reshape(features_a, (-1, 1))
    features_b = np.reshape(features_b, (-1, 1))
    enc.fit(np.concatenate((features_a, features_b), axis=0))
    one_hot_a = enc.transform(features_a).toarray()
    one_hot_b = enc.transform(features_b).toarray()
    one_hot_names_a = []
    one_hot_names_b = []
    for c in categories:
        one_hot_names_a.append(feature_name_a + '_' + c)
        one_hot_names_b.append(feature_name_b + '_' + c)
    return one_hot_names_a, one_hot_a, one_hot_names_b, one_hot_b

def add_one_hot_features(features, one_hot_names, one_hot):
    one_hot_T = np.transpose(one_hot)
    for i in range(len(one_hot_names)):
        features[one_hot_names[i]] = one_hot_T[i]
    return features

def encode_labels(features):
    le = LabelEncoder()
    labels = le.fit_transform(features)
    return list(labels)

# Feature engineering
features = pd.DataFrame(data={})
features_devectored = pd.DataFrame(data={})

plant_area_categories = getCategories('Plant Area')
network_type_categories = getCategories('Network Type')

# Site
site_a = map_ips_to_col('Address A', 'Plant Area')
site_b = map_ips_to_col('Address B', 'Plant Area')
features_devectored['Site A'] = site_a
features_devectored['Site B'] = site_b

site_features = encode_one_hot_2('Site A', 'Site B', plant_area_categories, site_a, site_b)
features = add_one_hot_features(features, site_features[0], site_features[1])
features = add_one_hot_features(features, site_features[2], site_features[3])

# Network Type
network_type_a = map_ips_to_col('Address A', 'Network Type')
network_type_b = map_ips_to_col('Address B', 'Network Type')
features_devectored['Network Type A'] = network_type_a
features_devectored['Network Type B'] = network_type_b

network_type_features = encode_one_hot_2('Network Type A', 'Network Type B', network_type_categories, network_type_a, network_type_b)
features = add_one_hot_features(features, network_type_features[0], network_type_features[1])
features = add_one_hot_features(features, network_type_features[2], network_type_features[3])

# Port
port_a = port_col_to_class(flow_data['Port A'])
port_b = port_col_to_class(flow_data['Port B'])
features_devectored['Port A'] = port_a
features_devectored['Port B'] = port_b

port_features = encode_one_hot_2('Port A', 'Port B', common_port_names, port_a, port_b)
features = add_one_hot_features(features, port_features[0], port_features[1])
features = add_one_hot_features(features, port_features[2], port_features[3])

# Same Site
asset_a_b = []
for i in range(len(site_a)):
    asset_a_b.append(1. if site_a[i] == site_b[i] else 0.)
features['Same Site'] = asset_a_b
features_devectored['Same Site'] = asset_a_b

# Same Subnet
address_a = list(flow_data['Address A'])
address_b = list(flow_data['Address B'])
subnet_a_b = []
for i in range(len(address_a)):
    subnet_a = '.'.join(address_a[i].split('.')[:-1])
    subnet_b = '.'.join(address_b[i].split('.')[:-1])
    subnet_a_b.append(1. if subnet_a == subnet_b else 0.)
features['Same Subnet'] = subnet_a_b
features_devectored['Same Subnet'] = subnet_a_b

# Site Pair
site_pair = []
for i in range(len(site_a)):
    site_pair.append(site_a[i] + '-' + site_b[i])
site_pair_labels = encode_labels(site_pair)
features['Site Pairs'] = site_pair_labels
features_devectored['Site Pairs'] = site_pair_labels

# Network Pair
network_pair = []
for i in range(len(network_type_a)):
    network_pair.append(network_type_a[i] + '-' + network_type_b[i])
network_pair_labels = encode_labels(network_pair)
features['Network Pairs'] = network_pair_labels
features_devectored['Network Pairs'] = network_pair_labels

# Save to file
features.to_csv(sys.argv[len(sys.argv) - 1], index=False)
features_devectored.to_csv(sys.argv[len(sys.argv) - 1].split('.')[0] + '-devectored.csv', index=False)

# Legacy features

# flow_data['Packets'] = clean_int_data(flow_data['Packets'])
# packets_z = normalize_col(flow_data['Packets'])
# features['Packets (z)'] = packets_z
#
# flow_data['Bytes'] = clean_int_data(flow_data['Bytes'])
# bytes_z = normalize_col(flow_data['Bytes'])
# features['Bytes (z)'] = bytes_z
#
# flow_data['Packets A -> B'] = clean_int_data(flow_data['Packets A -> B'])
# packets_ab_z = normalize_col(flow_data['Packets A -> B'])
# features['Packets A -> B (z)'] = packets_ab_z
#
# flow_data['Bytes A -> B'] = clean_int_data(flow_data['Bytes A -> B'])
# bytes_ab_z = normalize_col(flow_data['Bytes A -> B'])
# features['Bytes A -> B (z)'] = bytes_ab_z
#
# flow_data['Packets B -> A'] = clean_int_data(flow_data['Packets B -> A'])
# packets_ba_z = normalize_col(flow_data['Packets B -> A'])
# features['Packets B -> A (z)'] = packets_ba_z
#
# flow_data['Bytes B -> A'] = clean_int_data(flow_data['Bytes B -> A'])
# bytes_ba_z = normalize_col(flow_data['Bytes B -> A'])
# features['Bytes B -> A (z)'] = bytes_ba_z
#
# rel_start_z = normalize_col(flow_data['Rel Start'])
# features['Rel Start (z)'] = rel_start_z
#
# duration_z = normalize_col(flow_data['Duration'])
# features['Duration (z)'] = duration_z
#
# bit_rate_ab_z = normalize_col(flow_data['Bits/s A -> B'])
# features['Bits/s A -> B (z)'] = bit_rate_ab_z
#
# bit_rate_ba_z = normalize_col(flow_data['Bits/s B -> A'])
# features['Bits/s B -> A (z)'] = bit_rate_ba_z
