import model_decision_tree
import pandas as pd
import firewall_tags


def map_rule(name, threshold_decision):
    if '_' in name:
        operator = ' = ' if threshold_decision else ' != '
        index = name.index('_')
        return name[0:index] + operator + name[index + 1:]
    else:
        tcp = 'TCP'
        udp = 'UDP'
        if threshold_decision:
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
        left_child = tree_.children_left[node]
        right_child = tree_.children_right[node]
        rules = []
        rules += get_rules(model, feature_names, node=left_child, parent_name=name, parent_decision=False, parent_prev_rule=rule)
        rules += get_rules(model, feature_names, node=right_child, parent_name=name, parent_decision=True, parent_prev_rule=rule)
        return rules
    else:
        value = tree_.value[node]
        return [rule] if value[0][0] != 0 else []


def append_exclusive(array1, array2, el):
    if len(array1) == 0:
        for e in array2:
            array1.append(e)

    if el in array1:
        array1.remove(el)

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
        inner_label = list(list_tags.keys())[0]
        inner_list = list_tags[inner_label]
        return format_inner_list(inner_label, inner_list)
    else:
        formatted_tags = []
        for k, v in list_tags.items():
            formatted_tags.append(format_inner_list(k, v))
        return '(' + ' AND '.join(formatted_tags) + ')'


def format_inner_list(inner_label, inner_list):
    if len(inner_list) == 1:
        return 'tag ' + inner_label + ' = ' + inner_list[0]
    else:
        return '(' + ' OR '.join(list(map(lambda x: 'tag ' + inner_label + ' = ' + x, inner_list))) + ')'


def format_port_list(list_ports):
    if len(list_ports) == 0:
        return 'PORT all'
    elif len(list_ports) == 1:
        return 'PORT ' + str(list_ports[0])
    else:
        return '(' + ' AND '.join(list(map(lambda x: 'PORT ' + str(x), list_ports))) + ')'


def create_firewall_rules(rules, device_data, prefix_data, app_data):
    device_tags = firewall_tags.generate_tags(device_data)
    prefix_tags = firewall_tags.generate_tags(prefix_data)
    port_tags = firewall_tags.generate_tags(app_data)

    firewall_rules = []
    for r in rules:
        from_tags = {}
        to_tags = {}
        tcp_udp = 'both'
        ports = []

        for k in device_tags.keys():
            from_device_tags = []
            to_device_tags = []
            for s in r.split(' & '):
                if k in s:
                    if ' = ' in s:
                        if ' A ' in s:
                            from_device_tags = [s.split(' = ')[1]]
                        else:
                            to_device_tags = [s.split(' = ')[1]]
                    else:
                        if ' A ' in s:
                            from_device_tags = append_exclusive(from_device_tags, device_tags[k], s.split(' != ')[1])
                        else:
                            to_device_tags = append_exclusive(to_device_tags, device_tags[k], s.split(' != ')[1])
            if len(from_device_tags) > 0:
                from_tags[k] = from_device_tags
            if len(to_device_tags) > 0:
                to_tags[k] = to_device_tags

        for k in prefix_tags.keys():
            from_prefix_tags = []
            to_prefix_tags = []
            for s in r.split(' & '):
                if k in s:
                    if ' = ' in s:
                        if ' A ' in s:
                            from_prefix_tags = [s.split(' = ')[1]]
                        else:
                            to_prefix_tags = [s.split(' = ')[1]]
                    else:
                        if ' A ' in s:
                            from_prefix_tags = append_exclusive(from_prefix_tags, prefix_tags[k], s.split(' != ')[1])
                        else:
                            to_prefix_tags = append_exclusive(to_prefix_tags, prefix_tags[k], s.split(' != ')[1])
            if len(from_prefix_tags) > 0:
                from_tags[k] = from_prefix_tags
            if len(to_prefix_tags) > 0:
                to_tags[k] = to_prefix_tags

        for k in list(port_tags.keys())[1:]:
            for s in r.split(' & '):
                if k in s:
                    if ' = ' in s:
                        ports.append(s.split(' = ')[1])
                    else:
                        ports = append_exclusive(ports, port_tags[k], s.split(' != ')[1])
        for s in r.split(' & '):
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
        # if tcp_udp == 'both':
        #     firewall_rules.append('FROM ' + from_string + ' TO ' + to_string + ' ALLOW tcp ' + port_string)
        #     firewall_rules.append('FROM ' + from_string + ' TO ' + to_string + ' ALLOW udp ' + port_string)
        # else:
        #     firewall_rules.append('FROM ' + from_string + ' TO ' + to_string + ' ALLOW ' + tcp_udp + ' ' + port_string)
        firewall_rules.append('FROM ' + from_string + ' TO ' + to_string + ' ALLOW tcp ' + port_string)
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
    X_2 = pd.read_csv('data/features_fake.csv', header=0)
    models, feature_names = model_decision_tree.generate(X_0, X_1, X_2, 1)
    rules = get_rules(models[0], feature_names)
    print(*rules, sep='\n')
    device_data = pd.read_csv('data/devices.csv')
    prefix_data = pd.read_csv('data/prefixes.csv')
    app_data = pd.read_csv('data/applications.csv')
    firewall = create_firewall_rules(rules, device_data, prefix_data, app_data)
    print(*firewall, sep='\n')
