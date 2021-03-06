def generate_tags(data):
    tags = {}
    for c in data.columns[1:]:
        tags[c] = []
    for i, d in data.iterrows():
        for c in tags.keys():
            if d[c] not in tags[c]:
                tags[c].append(d[c])
    return tags
