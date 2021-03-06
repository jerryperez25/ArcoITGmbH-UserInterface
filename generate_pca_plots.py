import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import os
import sys

def perform_pca(data, n_components=2):
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(data)
    return pca

def create_pca_scatterplot(pca, plot_name):
    plt.figure()
    x = pca.components_[0]
    y = pca.components_[1]
    # p_colors = []
    # for i in range(len(x)):
    #     x_i = x[i]
    #     y_i = y[i]
    #     p_color = None
    #     if x_i < 0.5:
    #         if y_i < -0.2:
    #             p_color = [0.4, 0.4, 1]
    #         elif y_i >= -0.2 and y_i < 0.1:
    #             p_color = [0.2, 0.2, 1]
    #         else:
    #             p_color = [0, 0, 1]
    #     else:
    #         p_color = [1, 0.2, 0.2]
    #     p_colors.append(p_color)
    # plt.scatter(x, y, c=p_colors)
    plt.scatter(x, y)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Scatterplot')
    plt.savefig('output/' + plot_name + 'scatterplot.png')
    plt.close()

def create_scree_plot(pca, plot_name):
    plt.figure()
    plt.plot(np.arange(1, data.shape[1] + 1), pca.singular_values_, 'bo-')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Singular Value')
    plt.title('Scree Plot')
    plt.savefig('output/' + plot_name + 'scree.png')
    plt.close()

# Get arguments
file_path = 'output/features.csv'
if len(sys.argv) > 1:
    file_path = sys.argv[1]
plot_name_prefix = 'pca_'
if len(sys.argv) > 2:
    plot_name_prefix = 'pca_' + sys.argv[2] + '_'

# Features
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

# Create output directory if DNE
if not os.path.exists('output'):
    os.makedirs('output')

# Perform PCA to obtain just 2 principal components
pca = perform_pca(data, n_components=2)

# Plot the two principle components
create_pca_scatterplot(pca, plot_name_prefix)

# Perform PCA with components equal to number of features
pca = perform_pca(data, n_components=data.shape[1])

# Plot singular values
create_scree_plot(pca, plot_name_prefix)
