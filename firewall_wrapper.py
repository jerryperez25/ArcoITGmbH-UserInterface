from time import sleep
import pandas as pd
import generate_features
import gan_flow_data
import generate_gan_feature_data
import feature_faker
import model_decision_tree
import firewall_rules

delay_time = 1


def create_firewall_rules(device_data_1, prefix_data_1, app_data_1, flow_data_1):
    print('Mapping tags onto flow data')
    device_data = pd.read_csv('data/devices.csv')
    prefix_data = pd.read_csv('data/prefixes.csv')
    app_data = pd.read_csv('data/applications.csv')
    flow_data = pd.read_csv('data/flow.csv')
    features_0 = generate_features.generate(device_data, prefix_data, app_data, flow_data)
    print('Mapping completed')
    sleep(delay_time)

    print('Training the GAN on the flow data')
    generator, headers = gan_flow_data.generate(features_0)
    print('Training completed')
    sleep(delay_time)

    print('Generating fake flow data')
    features_1 = generate_gan_feature_data.generate(generator, headers, flow_data.shape[0] / 2)
    features_2 = feature_faker.generate(features_0, flow_data.shape[0] / 2)
    print('Generating completed')
    sleep(delay_time)

    print('Training the machine learning model')
    models, feature_names = model_decision_tree.generate(features_0, features_1, features_2, 2)
    print('Training completed')
    sleep(delay_time)

    print('Converting model into firewall rules')
    rules = firewall_rules.generate(models, feature_names, device_data, prefix_data, app_data)
    print('Converting completed')
    sleep(delay_time)
    return rules
