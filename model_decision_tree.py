import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def create_X_y(X_0, X_1):
    X = pd.concat((X_0, X_1), axis=0)
    y = []
    for _ in range(len(X_0)):
        y.append(0)
    for _ in range(len(X_1)):
        y.append(1)
    return X, y


def train_test_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y0 = model.predict(X_test)

    acc_count = 0
    for i in range(len(y0)):
        acc_count += 1 if y0[i] == y_test[i] else 0
    accuracy = acc_count / len(y_test)
    return model, accuracy


def generate(X_0, X_1, N_trees):
    X, y = create_X_y(X_0, X_1)
    feature_names = X.columns.tolist()

    models = []
    for _ in range(N_trees):
        model, accuracy = train_test_model(X, y)
        models.append(model)
    return models, feature_names


if __name__ == "__main__":
    X_0 = pd.read_csv('data/features.csv')
    X_1 = pd.read_csv('data/features_gan.csv')
    X, y = create_X_y(X_0, X_1)
    feature_names = X.columns.tolist()
    model, accuracy = train_test_model(X, y)
    print('Accuracy: ' + str(accuracy * 100) + '%')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=300)
    tree.plot_tree(model, feature_names=feature_names, class_names=['Good', 'Bad'], filled=True)
    fig.savefig('output/decision_tree.png')
