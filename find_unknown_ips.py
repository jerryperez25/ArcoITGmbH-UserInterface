import pandas as pd

flow_data = pd.read_csv('sample_data/sample_data.csv', header=0)
ip_data = pd.read_csv('sample_data/ip_inventory.csv', header=0)
ip_list = ip_data['LAN IP Address']

def find_unknowns(address_name, unknowns):
    for ip in flow_data[address_name]:
        if not ip in ip_list.values:
            if ip in unknowns:
                unknowns[ip] = unknowns[ip] + 1
            else:
                unknowns[ip] = 1

unknown_ips = {}
find_unknowns('Address A', unknown_ips)
find_unknowns('Address B', unknown_ips)

output_data = []
for k, v in unknown_ips.items():
    output_data.append([k, v])
output_data = pd.DataFrame(output_data, columns=['IP Address', 'Occurrences'])
output_data = output_data.sort_values(by=['Occurrences'], ascending=False)
output_data.to_csv('output/unknown_ips.csv', index=False)
