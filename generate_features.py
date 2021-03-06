import firewall_tags
import pandas as pd

ADDRESS_A_COL = 0
PORT_A_COL = 1
ADDRESS_B_COL = 2
PORT_B_COL = 3


def get_index(df, column, value, compare_func=lambda a, b: a == b):
    for i, d in df.iterrows():
        if compare_func(d[column], value):
            return i
    return -1


def compare_prefix(prefix, ip):
    prefix_subnet = prefix.split('.')[0:-1]
    prefix_range = prefix.split('.')[-1]
    prefix_range_min = int(prefix_range.split('/')[0])
    prefix_range_max = int(prefix_range.split('/')[1])
    ip_subnet = ip.split('.')[0:-1]
    ip_broadcast = int(ip.split('.')[-1])
    return ip_subnet == prefix_subnet and prefix_range_min <= ip_broadcast <= prefix_range_max


def map_ips(a_b, col, flow, data, tags, compare_func=None):
    devices_header = []
    devices = []
    for n in tags.keys():
        names = tags[n]
        for name in names:
            devices_header.append(n + ' ' + a_b + '_' + name)
        for ip in flow[flow.columns[col]]:
            i = get_index(data, 0, ip) if compare_func is None else get_index(data, 0, ip, compare_func)
            one_hot = [0] * len(names)
            if i != -1:
                v = data[n][i]
                one_hot[names.index(v)] = 1
            devices.append(one_hot)
    return pd.DataFrame(devices, columns=devices_header)


def map_ports(a_b, col, flow, port_data, tags):
    tag_name = list(tags.keys())[1]
    names = tags[tag_name]
    tcp_udp_name = list(tags.keys())[0]

    ports_headers = []
    ports = []
    tcp_udp = []
    for name in names:
        ports_headers.append(tag_name + ' ' + a_b + '_' + name)
    for port in flow[flow.columns[col]]:
        i = get_index(port_data, 0, port)

        one_hot_ports = [0] * len(names)
        one_hot_tcp_udp = [0] * 2
        if i != -1:
            v_port = port_data[tag_name][i]
            one_hot_ports[names.index(v_port)] = 1
            v_tcp_udp = port_data[tcp_udp_name][i]
            one_hot_tcp_udp[0 if v_tcp_udp == 'TCP' else 1] = 1
        ports.append(one_hot_ports)
        tcp_udp.append(one_hot_tcp_udp)
    df_ports = pd.DataFrame(ports, columns=ports_headers)
    df_tcp_udp = pd.DataFrame(tcp_udp, columns=['TCP', 'UDP'])
    return pd.concat([df_ports, df_tcp_udp], axis=1)


def generate(device_data, prefix_data, app_data, flow_data):
    device_tags = firewall_tags.generate_tags(device_data)
    prefix_tags = firewall_tags.generate_tags(prefix_data)
    port_tags = firewall_tags.generate_tags(app_data)

    devices_a = map_ips('A', ADDRESS_A_COL, flow_data, device_data, device_tags)
    devices_b = map_ips('B', ADDRESS_B_COL, flow_data, device_data, device_tags)

    prefixes_a = map_ips('A', ADDRESS_A_COL, flow_data, prefix_data, prefix_tags, compare_prefix)
    prefixes_b = map_ips('B', ADDRESS_B_COL, flow_data, prefix_data, prefix_tags, compare_prefix)

    ports_b = map_ports('B', PORT_B_COL, flow_data, app_data, port_tags)

    return pd.DataFrame(pd.concat([devices_a, devices_b, prefixes_a, prefixes_b, ports_b], axis=1))


if __name__ == "__main__":
    device_data = pd.read_csv('data/devices.csv')
    prefix_data = pd.read_csv('data/prefixes.csv')
    app_data = pd.read_csv('data/applications.csv')
    flow_data = pd.read_csv('data/flow2.csv')
    features = generate(device_data, prefix_data, app_data, flow_data)
    features.to_csv('data/features.csv', index=False)
