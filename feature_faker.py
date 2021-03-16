import numpy as np
import pandas as pd


def generate(features, n_samples):
    n_one_hot = []
    headers = features.columns

    prev_one_hot = ''
    for h in headers:
        if '_' in h:
            one_hot = h.split('_')[0]
            if one_hot == prev_one_hot:
                n_one_hot[-1] += 1
            else:
                prev_one_hot = one_hot
                n_one_hot.append(1)
    n_one_hot.append(2)

    data = []
    for i in range(int(n_samples)):
        r0 = np.random.randint(0, features.shape[0])
        r_feature = features.iloc[r0]

        weights = []
        for i in range(len(n_one_hot)):
            if i < 2:
                weights.append(2)
            elif i == len(n_one_hot) - 2:
                weights.append(5)
            elif i == len(n_one_hot) - 1:
                weights.append(2)
            else:
                weights.append(1)
        weights_sum = sum(weights)
        for i in range(len(weights)):
            weights[i] /= weights_sum

        r1 = np.random.choice(len(n_one_hot), 1, p=weights)[0]
        col_i = 0
        for i in range(r1):
            col_i += n_one_hot[i]

        curr_value = 0
        for i in range(n_one_hot[r1]):
            if r_feature[col_i + i] == 1:
                curr_value = i

        r2 = curr_value
        while r2 == curr_value:
            if n_one_hot[r1] != 1:
                r2 = np.random.randint(0, n_one_hot[r1])
            else:
                break
        curr_value += col_i
        col_i += r2

        r_feature[curr_value] = 0
        r_feature[col_i] = 1
        data.append(r_feature)

    return pd.DataFrame(data, columns=features.columns)


if __name__ == "__main__":
    features_0 = pd.read_csv('data/features.csv')
    features_1 = generate(features_0, features_0.shape[0] / 2)
    features_1.to_csv('data/features_fake.csv', index=False)
