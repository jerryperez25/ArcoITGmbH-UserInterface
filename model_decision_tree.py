import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def generate_model(features_c0, features_c1):
    X_0 = pd.DataFrame(pd.read_csv(features_c0, header=0))
    X_1 = pd.DataFrame(pd.read_csv(features_c1, header=0))
    X = pd.concat((X_0, X_1), axis=0)
    y = []
    for _ in range(len(X_0)):
        y.append(0)
    for _ in range(len(X_1)):
        y.append(1)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = DecisionTreeClassifier(random_state=0, max_depth=3)
    model.fit(X_train, y_train)

    y0 = model.predict(X_test)

    acc_count = 0
    for i in range(len(y0)):
        acc_count += 1 if y0[i] == y_test[i] else 0
    accuracy = acc_count / len(y_test)
    print('Accuracy: ' + str(accuracy * 100) + '%')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=300)
    tree.plot_tree(model, feature_names=feature_names, class_names=['Good', 'Bad'], filled=True)
    fig.savefig('output/decision_tree.png')

    return model, feature_names
