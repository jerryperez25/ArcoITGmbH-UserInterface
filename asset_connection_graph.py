import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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

address_a = map_ips_to_col('Address A', 'Plant Area')
address_b = map_ips_to_col('Address B', 'Plant Area')
port_a = flow_data['Port A']
port_b = flow_data['Port B']

data = pd.concat([address_a, port_a, address_b, port_b], 1)
data = data.dropna()

print('Data loaded')

graph = {}
for i, d in data.iterrows():
    if d['Address A'] in graph:
        connections = graph[d['Address A']]
        if d['Address B'] in connections:
            ports = connections[d['Address B']]
            if not d['Port A'] in ports:
                ports.append(d['Port A'])
            if not d['Port B'] in ports:
                ports.append(d['Port B'])
        else:
            connections[d['Address B']] = [d['Port A'], d['Port B']]
    else:
        graph[d['Address A']] = {d['Address B']: [d['Port A'], d['Port B']]}

print('Internal graph created')

vis_graph = nx.DiGraph()
edge_labels = {}
for k0, v0 in graph.items():
    for k1, v1 in v0.items():
        if k0 != k1:
            common_values = list(filter(lambda v: v <= 1024, v1))
            if len(common_values) > 0:
                vis_graph.add_edge(k1, k0)
                edge_labels[(k0, k1)] = ', '.join(map(str, common_values))

def formate_node_label(label):
    if label == 'Other ProcN Devices':
        return 'Other ProcN\n Devices'
    if label == 'EAF Dust Washing':
        return 'EAF Dust\nWashing'
    return label

pos = nx.circular_layout(vis_graph)
fig = plt.figure()
nx.draw(vis_graph, pos, node_size=5000, node_color=[0.9,0.9,0.9], labels={node:formate_node_label(str(node)) for node in vis_graph.nodes()}, font_size=10, edgecolors=['black'])
nx.draw_networkx_edge_labels(vis_graph, pos, edge_labels=edge_labels)

print('Plotted graph')

plt.show()
