#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:12:21 2017

@author: codeplay2017
"""

from time import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from data import data_factory


def main(istrain=True):
    # Display progress logs on stdout
#    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    ###############################################################################
    # load the data on disk and load it as numpy arrays
    
    train_speed=[10,20,30,40,50]
#    test_speed=[50]
    
    data_dir = '/home/codeplay2018/E/code/lab/data/TestDataFromWen/arranged/steady_condition/pkl'
    print('[INFO] importing data...')
    data_fn = data_factory.get_data_from('wen')
    trainset, validset, testset = data_fn(data_dir,
                                         speed_list=train_speed,
                                         train_split=0.8,
                                         vt_split=0.5,
                                         divide_step=20,
                                         data_length=8192,
                                         fft=True,
                                         normalize=False,
                                         verbose=False, use_speed=False)
#    adavalidset, adatestset, adatestset2 = data_fn(data_dir,
#                                         speed_list=test_speed,
#                                         train_split=0.9,
#                                         vt_split=0.5,
#                                         divide_step=5,
#                                         data_length =8192,
#                                         fft = True,
#                                         normalize=False,
#                                         verbose=False, use_speed=False)
#    adatestset.join_data(adatestset2)
#    print('adavalidset and adatestset %d, %d'%(adavalidset.num_examples(),
#                                               adatestset.num_examples()))
    
    # introspect the images arrays to find the shapes (for plotting)
    n_samples, image_l = trainset.images.shape
    
    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X_train = trainset.images
    X_test = testset.images
    n_features = X_train.shape[1]
    
    # the label to predict is the id of the person
    y_train = np.where(trainset.labels==1)[1]
    y_test = np.where(testset.labels==1)[1]
    target_names = np.unique(y_train)
    n_classes = target_names.shape[0]
    
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
  
    ##########################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 50
    
    print("Extracting the top %d PC from %d features"
          % (n_components, X_train.shape[1]))
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    print('information ratio is '+str(pca.explained_variance_ratio_))
    print("pca established in %0.3fs" % (time() - t0))
    
#    eigenfaces = pca.components_.reshape((n_components, l))
    
    print("Projecting the input data on the PCs orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("pca done in %0.3fs" % (time() - t0))
    
    
    ###########################################################################
    # Train a SVM classification model
    
    print("Fitting the classifier to the training set")
    kernel = 'rbf'
    t0 = time()
    param_grid_rbf = {'C': [1e2],
                  'gamma': [0.005]}
    param_grid_linear = {'C': [1e2]}
    param_grid_map = {'linear': param_grid_linear,
                      'rbf': param_grid_rbf}
    param_grid = param_grid_map[kernel]
    if istrain:
        clf = GridSearchCV(SVC(kernel='linear'), param_grid)
        clf = clf.fit(X_train_pca, y_train)
        with open('clf.pkl','wb') as f:
            pickle.dump(clf, f)
    else:    
        with open('clf.pkl', 'rb') as f:
            clf = pickle.load(f)
    print("training done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    
    
    ###########################################################################
    # Quantitative evaluation of the model quality on the test set
    
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("prediction done in %0.3fs" % (time() - t0))
    print(type(y_pred))
    
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    return y_pred, y_test
    
#    prediction_titles = [title(y_pred, y_test, target_names, i)
#                     for i in range(y_pred.shape[0])]
#
#    plot_gallery(X_test, prediction_titles, h, w)
#    
#    # plot the gallery of the most significative eigenfaces
#    
#    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#    plot_gallery(eigenfaces, eigenface_titles, h, w)
#    
#    plt.show()


###############################################################################
# Qualitative evaluation of the predictions using matplotlib
    

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


if __name__ == '__main__':
    yp,yt = main(istrain=True)



















