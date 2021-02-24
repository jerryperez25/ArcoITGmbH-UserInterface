import model_decision_tree


def map_rule(name, threshold_decision):
    if '_' in name:
        operator = ' = ' if not threshold_decision else ' != '
        index = name.index('_')
        return name[0:index] + operator + name[index + 1:]
    elif name == 'Same Site':
        return 'Site A = Site B'
    elif name == 'Unique Connection':
        return 'Unique Connection'


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


model, feature_names = model_decision_tree.generate_model('output/features.csv', 'output/features_random.csv')
print(get_rules(model, feature_names))
