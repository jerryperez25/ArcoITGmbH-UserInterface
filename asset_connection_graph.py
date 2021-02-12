import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')

common_port_cutoff = 1024

flow_data = pd.read_csv('sample_data/sample_data.csv', header=0)
ip_data = pd.read_csv('sample_data/ip_inventory.csv', header=0, index_col=0)

def map_ips_to_col(address_name, col_name):
    site_list = []
    for ip in flow_data[address_name]:
        try:
            name = ip_data.at[ip, col_name]
            site_list.append(name)
        except:
            site_list.append(pd.NaT)
    return pd.DataFrame(site_list, columns=[address_name])

site_a = map_ips_to_col('Address A', 'Plant Area')
site_b = map_ips_to_col('Address B', 'Plant Area')
port_a = flow_data['Port A']
port_b = flow_data['Port B']

data = pd.concat([site_a, port_a, site_b, port_b], 1)
data = data.dropna()

print('Data loaded')

main_graph = {}
for i, d in data.iterrows():
    ports = []
    if d['Port A'] <= common_port_cutoff:
        ports.append(d['Port A'])
    if d['Port B'] <= common_port_cutoff:
        ports.append(d['Port B'])
    if d['Address A'] in main_graph:
        connections = main_graph[d['Address A']]
        if d['Address B'] in connections:
            used_ports = connections[d['Address B']]
            for p in ports:
                if not p in used_ports:
                    used_ports.append(p)
        elif len(ports) > 0:
            connections[d['Address B']] = ports
    elif len(ports) > 0:
        main_graph[d['Address A']] = {d['Address B']: ports}

print('Main graph created')

inter_graph = {}
ip_a = flow_data['Address A']
ip_b = flow_data['Address B']
for i in range(site_a.shape[0]):
    site_a_i = site_a['Address A'][i]
    ports = []
    if port_a[i] <= common_port_cutoff:
        ports.append(port_a[i])
    if port_b[i] <= common_port_cutoff:
        ports.append(port_b[i])
    if site_a_i == site_b['Address B'][i]:
        ip_a_i = ip_a[i]
        ip_b_i = ip_b[i]
        if site_a_i in inter_graph:
            if ip_a_i in inter_graph[site_a_i]:
                if ip_b_i in inter_graph[site_a_i][ip_a_i]:
                    used_ports = inter_graph[site_a_i][ip_a_i][ip_b_i]
                    for p in ports:
                        if not p in used_ports:
                            used_ports.append(p)
                elif len(ports) > 0:
                    inter_graph[site_a_i][ip_a_i][ip_b_i] = ports
            elif len(ports) > 0:
                inter_graph[site_a_i][ip_a_i] = {ip_b_i: ports}
        elif len(ports) > 0:
            inter_graph[site_a_i] = {ip_a_i: {ip_b_i: ports}}

print('Internal graphs created')

def formate_node_label(label):
    if label == 'Other ProcN Devices':
        return 'Other ProcN\n Devices'
    if label == 'EAF Dust Washing':
        return 'EAF Dust\nWashing'
    return label

def plot_graph(graph_name, graph, fig_size):
    vis_graph = nx.DiGraph()
    edge_labels = {}
    for k0, v0 in graph.items():
        for k1, v1 in v0.items():
            if k0 != k1:
                if len(v1) > 0:
                    vis_graph.add_edge(k1, k0)
                    edge_labels[(k0, k1)] = ', '.join(map(str, v1))

    pos = nx.kamada_kawai_layout(vis_graph)
    fig = plt.figure(figsize=fig_size, dpi=300)
    nx.draw(vis_graph, pos, node_size=5500, node_color=[0.9,0.9,0.9], labels={node:formate_node_label(str(node)) for node in vis_graph.nodes()}, font_size=10, edgecolors='black')
    nx.draw_networkx_edge_labels(vis_graph, pos, edge_labels=edge_labels)

    plt.savefig('output/asset_connection_' + graph_name + '.png')

plot_graph('Main', main_graph, (9, 9))
print('Plotted main graph')

for site, connections in inter_graph.items():
    plot_graph(site, connections, (7, 7))
    print('Plotted ' + site + ' internal graph')
