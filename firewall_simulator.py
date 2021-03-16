import generate_features
import pandas as pd

device_data = None
prefix_data = None
app_data = None
rules = None


def init(device_data_, prefix_data_, app_data_, rules_):
    global device_data
    device_data = device_data_
    global prefix_data
    prefix_data = prefix_data_
    global app_data
    app_data = app_data_
    global rules
    rules = rules_


def map_flow(flow):
    mapped_flow = generate_features.generate(device_data, prefix_data, app_data, pd.DataFrame([flow]))
    from_tags = {}
    to_tags = {}
    port_tag = flow[3]
    for m in mapped_flow.columns[:-2]:
        if mapped_flow[m][0] == 1:
            m_split = m.split('_')
            tag_name = m_split[0]
            tag_value = m_split[1]
            if tag_name[-2:] == ' A':
                tag_name = tag_name[:-2]
                if tag_name in from_tags:
                    from_tags[tag_name].append(tag_value)
                else:
                    from_tags[tag_name] = [tag_value]
            else:
                tag_name = tag_name[:-2]
                if tag_name in to_tags:
                    to_tags[tag_name].append(tag_value)
                else:
                    to_tags[tag_name] = [tag_value]
    tcp_udp = 'tcp' if mapped_flow['TCP'][0] == 1 else 'udp'
    return {'FROM': from_tags, 'TO': to_tags, 'TCP/UDP': tcp_udp, 'PORT': port_tag}


def parse_section(section):
    tags = {}
    if section != 'any':
        section = section.replace('(', '')
        section = section.replace(')', '')
        section = section.replace('tag ', '')

        and_sections = section.split(' AND ')
        for and_section in and_sections:
            or_sections = and_section.split(' OR ')
            for or_section in or_sections:
                or_split = or_section.split(' = ')
                or_name = or_split[0]
                or_value = or_split[1]
                if or_name in tags:
                    tags[or_name].append(or_value)
                else:
                    tags[or_name] = [or_value]
    return tags


def parse_rule(rule):
    allow_deny = 'ALLOW' if 'ALLOW' in rule else 'DENY'
    from_rule = parse_section(rule[rule.index('FROM') + 5:rule.index('TO') - 1])
    to_rule = parse_section(rule[rule.index('TO') + 3:rule.index(allow_deny) - 1])
    tcp_udp = rule[rule.index(allow_deny) + len(allow_deny) + 1:rule.index(allow_deny) + len(allow_deny) + 4]
    port_rule = rule[rule.index(allow_deny) + len(allow_deny) + 5:]
    return {'ALLOW/DENY': allow_deny, 'FROM': from_rule, 'TO': to_rule, 'TCP/UDP': tcp_udp, 'PORT': port_rule}


def apply_rule(parsed_rule, mapped_flow):
    allow_deny = parsed_rule['ALLOW/DENY']
    from_rule = parsed_rule['FROM']
    to_rule = parsed_rule['TO']
    tcp_udp_rule = parsed_rule['TCP/UDP']
    port_rule = parsed_rule['PORT']

    from_tags = mapped_flow['FROM']
    to_tags = mapped_flow['TO']
    tcp_udp_tag = mapped_flow['TCP/UDP']
    port_tag = mapped_flow['PORT']

    from_result = True
    for r, tags in from_rule.items():
        if r in from_tags:
            from_result = False
            for t in tags:
                if t in from_tags[r]:
                    from_result = True
                    break
            if not from_result:
                break
        else:
            from_result = False
            break

    to_result = True
    for r, tags in to_rule.items():
        if r in to_tags:
            to_result = False
            for t in tags:
                if t in to_tags[r]:
                    to_result = True
                    break
            if not to_result:
                break
        else:
            to_result = False
            break

    tcp_udp_result = tcp_udp_rule == tcp_udp_tag

    port_split = port_rule.replace('(', '').replace(')', '').split(' AND ')
    port_result = port_rule == 'PORT all' or 'PORT ' + port_tag in port_split

    if from_result and to_result and tcp_udp_result and port_result:
        return allow_deny

    return 'NA'

def simulate(flow):
    if device_data is not None and prefix_data is not None and app_data is not None and rules is not None:
        mapped_flow = map_flow(flow)
        for r in rules:
            parsed_rule = parse_rule(r)
            result = apply_rule(parsed_rule, mapped_flow)
            if result != 'NA':
                return result
        return 'DENY'
    else:
        return 'ERROR'


if __name__ == "__main__":
    import pandas as pd
    import model_decision_tree
    import firewall_rules

    d_data = pd.read_csv('data/devices.csv')
    p_data = pd.read_csv('data/prefixes.csv')
    a_data = pd.read_csv('data/applications.csv')
    X_0 = pd.read_csv('data/features.csv', header=0)
    X_1 = pd.read_csv('data/features_gan.csv', header=0)
    X_2 = pd.read_csv('data/features_fake.csv', header=0)
    models, feature_names = model_decision_tree.generate(X_0, X_1, X_2, 1)
    f_rules = firewall_rules.generate(models, feature_names, d_data, p_data, a_data)
    print(*f_rules, sep='\n')

    init(d_data, p_data, a_data, f_rules)
    result_pos = simulate(['10.12.0.252', '1547', '10.12.0.39', '80'])
    print('Positive result: ', result_pos)
    result_neg = simulate(['10.12.0.252', '60896', '10.12.1.24', '502'])
    print('Negative result: ', result_neg)
