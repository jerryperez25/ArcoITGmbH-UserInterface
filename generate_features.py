import firewall_tags
import pandas as pd

ADDRESS_A_COL = 0
PORT_A_COL = 1
ADDRESS_B_COL = 2
PORT_B_COL = 3


def get_row(df, index):
    try:
        return df.loc[[index]]
    except KeyError:
        return None


def map_ips(a_b, col, flow, data, tags):
    devices_header = []
    devices = []
    for n in tags.keys():
        names = tags[n]
        for name in names:
            devices_header.append(n + ' ' + a_b + '_' + name)
        for ip in flow[flow.columns[col]]:
            r = get_row(data, ip)
            one_hot = [0] * len(names)
            if r is not None:
                one_hot[names.index(r[n][0])] = 1
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
        r = get_row(port_data, port)

        one_hot_ports = [0] * len(names)
        one_hot_tcp_udp = [0] * 2
        if r is not None:
            v_port = r[tag_name]
            one_hot_ports[names.index(v_port)] = 1
            v_tcp_udp = r[tcp_udp_name]
            one_hot_tcp_udp[0 if v_tcp_udp == 'TCP' else 1] = 1
        else:
            one_hot_tcp_udp[0] = 1
        ports.append(one_hot_ports)
        tcp_udp.append(one_hot_tcp_udp)
    df_ports = pd.DataFrame(ports, columns=ports_headers)
    df_tcp_udp = pd.DataFrame(tcp_udp, columns=['TCP', 'UDP'])
    return pd.concat([df_ports, df_tcp_udp], axis=1)


def expand_prefixes(data):
    expanded_data = []
    for i, r in data.iterrows():
        prefix = r[0]
        prefix_subnet = prefix.split('.')[:-1]
        prefix_range = prefix.split('.')[-1]
        prefix_range_min = int(prefix_range.split('/')[0])
        prefix_range_max = int(prefix_range.split('/')[1])
        for broadcast in range(prefix_range_min, prefix_range_max + 1):
            ip_array = prefix_subnet + [str(broadcast)]
            ip = '.'.join(ip_array)
            expanded_data.append([ip, r[1]])
    df = pd.DataFrame(expanded_data, columns=data.columns)
    return df.set_index(df.columns[0])


def generate(device_data, prefix_data, app_data, flow_data):
    device_tags = firewall_tags.generate_tags(device_data)
    prefix_tags = firewall_tags.generate_tags(prefix_data)
    port_tags = firewall_tags.generate_tags(app_data)

    device_data = device_data.set_index(device_data.columns[0])

    devices_a = map_ips('A', ADDRESS_A_COL, flow_data, device_data, device_tags)
    devices_b = map_ips('B', ADDRESS_B_COL, flow_data, device_data, device_tags)

    expanded_prefix_data = expand_prefixes(prefix_data)

    prefixes_a = map_ips('A', ADDRESS_A_COL, flow_data, expanded_prefix_data, prefix_tags)
    prefixes_b = map_ips('B', ADDRESS_B_COL, flow_data, expanded_prefix_data, prefix_tags)

    ports_b = map_ports('B', PORT_B_COL, flow_data, app_data, port_tags)

    return pd.DataFrame(pd.concat([devices_a, devices_b, prefixes_a, prefixes_b, ports_b], axis=1))


if __name__ == "__main__":
    device_data = pd.read_csv('data/devices.csv')
    prefix_data = pd.read_csv('data/prefixes.csv')
    app_data = pd.read_csv('data/applications.csv')
    flow_data = pd.read_csv('data/flow.csv')
    features = generate(device_data, prefix_data, app_data, flow_data)
    features.to_csv('data/features.csv', index=False)
