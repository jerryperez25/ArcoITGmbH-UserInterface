#/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition

def perform_pca(data, n_components=2):
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(data)
    return pca

def create_pca_scatterplot(pca):
    plt.figure()
    plt.plot(pca.components_[0], pca.components_[1], 'o')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Scatterplot')
    plt.savefig('pca_scatterplot.png')
    plt.close()
    
def create_scree_plot(pca):
    plt.figure()
    plt.plot(np.arange(1, data.shape[0] + 1), pca.singular_values_, 'bo-')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Singular Value')
    plt.title('Scree Plot')
    plt.savefig('pca_scree.png')
    plt.close()

###
# INSERT CODE TO LOAD YOUR DATA HERE
# - Ensure that all features are standardized
###
data = np.random.randint(100, size=(15, 100))
###

# Perform PCA to obtain just 2 principal components
pca = perform_pca(data, n_components=2)

# Plot the two principle components
create_pca_scatterplot(pca)

# Perform PCA with components equal to number of features
pca = perform_pca(data, n_components=data.shape[0])

# Plot singular values
create_scree_plot(pca)
