import generate_features
import gan_flow_data
import generate_gan_feature_data
import model_decision_tree
import firewall_rules


def create_firewall_rules(device_data, prefix_data, app_data, flow_data):
    print('Starting features_0')
    features_0 = generate_features.generate(device_data, prefix_data, app_data, flow_data)
    print('Starting GAN')
    generator, headers = gan_flow_data.generate(features_0)
    print('Starting features_1')
    features_1 = generate_gan_feature_data.generate(generator, headers, flow_data.shape[0])
    print('Starting model')
    models, feature_names = model_decision_tree.generate(features_0, features_1, 10)
    print('Starting rules')
    rules = firewall_rules.generate(models, feature_names, device_data, prefix_data, app_data)
    return rules
