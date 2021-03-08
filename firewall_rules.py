import model_decision_tree
import pandas as pd
import firewall_tags


def map_rule(name, threshold_decision):
    if '_' in name:
        operator = ' = ' if not threshold_decision else ' != '
        index = name.index('_')
        return name[0:index] + operator + name[index + 1:]
    else:
        tcp = 'TCP'
        udp = 'UDP'
        if not threshold_decision:
            return 'ALLOW ' + name
        else:
            return 'ALLOW ' + (tcp if name == udp else udp)


def get_rules(model, feature_names, node=0, parent_name=None, parent_decision=False, parent_prev_rule=None):
    parent_rule = map_rule(str(parent_name), parent_decision) if parent_name != None else None
    rule = parent_prev_rule + ' & ' + parent_rule if parent_prev_rule != None else parent_rule
    tree_ = model.tree_
    feature = tree_.feature[node]
    if feature >= 0:
        name = feature_names[feature]
        threshold = tree_.threshold[node]
        left_child = tree_.children_left[node]
        right_child = tree_.children_right[node]
        rules = []
        rules += get_rules(model, feature_names, node=left_child, parent_name=name, parent_decision=False, parent_prev_rule=rule)
        rules += get_rules(model, feature_names, node=right_child, parent_name=name, parent_decision=True, parent_prev_rule=rule)
        return rules
    else:
        value = tree_.value[node]
        return [rule] if value[0][0] != 0 else []


def add_all_except(array1, array2, el, tag=None):
    for e in array2:
        if e != el and e not in array1:
            if tag is not None:
                array1.append((tag, e))
            else:
                array1.append(e)
    return array1


def map_ports(list_tags, app_data):
    list_ports = []
    for t in list_tags:
        ports = app_data.loc[app_data['Application Name'] == t]
        for p in ports['Port']:
            list_ports.append(p)
    return list_ports


def format_tag_list(list_tags):
    if len(list_tags) == 0:
        return 'any'
    elif len(list_tags) == 1:
        return 'tag ' + list_tags[0][0] + ' = ' + list_tags[0][1]
    else:
        return '(' + ' AND '.join(list(map(lambda x: 'tag ' + x[0] + ' = ' + x[1], list_tags))) + ')'


def format_port_list(list_ports):
    if len(list_ports) == 0:
        return 'PORT = all'
    elif len(list_ports) == 1:
        return 'PORT ' + str(list_ports[0])
    else:
        return '(' + ' AND '.join(list(map(lambda x: 'PORT ' + str(x), list_ports))) + ')'


def create_firewall_rules(rules, device_data, prefix_data, app_data):
    device_tags = firewall_tags.generate_tags(device_data)
    prefix_tags = firewall_tags.generate_tags(prefix_data)
    port_tags = firewall_tags.generate_tags(app_data)

    # Should keep track of both inclusions and exclusions for from, to, and ports (not only inclusions)
    firewall_rules = []
    for r in rules:
        from_tags = []
        to_tags = []
        tcp_udp = 'both'
        ports = []
        for s in r.split(' & '):
            for k in device_tags.keys():
                if k in s:
                    if ' = ' in s:
                        from_tags.append((k, s.split(' = ')[1]))
                    else:
                        pass
                        # from_tags = add_all_except(from_tags, device_tags[k], s.split(' != ')[1], k)
            for k in prefix_tags.keys():
                if k in s:
                    if ' = ' in s:
                        from_tags.append((k, s.split(' = ')[1]))
                    else:
                        pass
                        # from_tags = add_all_except(from_tags, prefix_tags[k], s.split(' != ')[1], k)
            for k in list(port_tags.keys())[1:]:
                if k in s:
                    if ' = ' in s:
                        ports.append(s.split(' = ')[1])
                    else:
                        pass
                        # ports = add_all_except(ports, port_tags[k], s.split(' != ')[1])
            if 'TCP' in s and 'ALLOW' in s:
                if tcp_udp != 'udp':
                    tcp_udp = 'tcp'
                else:
                    tcp_udp = 'both'
            if 'UDP' in s and 'ALLOW' in s:
                if tcp_udp != 'tcp':
                    tcp_udp = 'udp'
                else:
                    tcp_udp = 'both'
        from_string = format_tag_list(from_tags)
        to_string = format_tag_list(to_tags)
        port_string = format_port_list(map_ports(ports, app_data))
        if tcp_udp == 'both':
            firewall_rules.append('FROM ' + from_string + ' TO ' + to_string + ' ALLOW tcp ' + port_string)
            firewall_rules.append('FROM ' + from_string + ' TO ' + to_string + ' ALLOW udp ' + port_string)
        else:
            firewall_rules.append('FROM ' + from_string + ' TO ' + to_string + ' ALLOW ' + tcp_udp + ' ' + port_string)
    return firewall_rules


def add_rules(rules_0, rules_1):
    for r in rules_1:
        if r not in rules_0:
            rules_0.append(r)
    return rules_0


def generate(models, feature_names, device_data, prefix_data, app_data):
    rules = []
    for model in models:
        add_rules(rules, get_rules(model, feature_names))
    firewall = create_firewall_rules(rules, device_data, prefix_data, app_data)
    return firewall


if __name__ == "__main__":
    X_0 = pd.read_csv('data/features.csv', header=0)
    X_1 = pd.read_csv('data/features_gan.csv', header=0)
    models, feature_names = model_decision_tree.generate(X_0, X_1, 1)
    rules = get_rules(models[0], feature_names)
    print(*rules, sep='\n')
    device_data = pd.read_csv('data/devices.csv')
    prefix_data = pd.read_csv('data/prefixes.csv')
    app_data = pd.read_csv('data/applications.csv')
    firewall = create_firewall_rules(rules, device_data, prefix_data, app_data)
    print(*firewall, sep='\n')
