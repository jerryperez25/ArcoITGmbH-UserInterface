import pandas as pd
import firewall_wrapper

if __name__ == "__main__":
    device_data = pd.read_csv('data/devices.csv')
    prefix_data = pd.read_csv('data/prefixes.csv')
    app_data = pd.read_csv('data/applications.csv')
    flow_data = pd.read_csv('data/flow.csv')
    firewall_rules = firewall_wrapper.create_firewall_rules(device_data, prefix_data, app_data, flow_data)

    print(*firewall_rules, sep='\n')
