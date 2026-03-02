#%% import packages 
import pandas as pd
import os
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



#%% load data 
with zipfile.ZipFile("ecg_data.zip","r") as zip_ref:
    zip_ref.extractall("ecg_data")

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'ecg_data/ecg_data.csv'), index_col=0)
    return data

data = load_data()

#%% defining useful functions 
def colorplot(clf, ax, x, y, h=100, precomputer=None):
    '''
    Overlay the decision areas as colors in an axes.

    Input:
        clf: trained classifier
        ax: axis to overlay color mesh on
        x: feature on x-axis
        y: feature on y-axis
        h(optional): steps in the mesh
    '''
    # Create a meshgrid the size of the axis
    xstep = (x.max() - x.min() ) / 20.0
    ystep = (y.max() - y.min() ) / 20.0
    x_min, x_max = x.min() - xstep, x.max() + xstep
    y_min, y_max = y.min() - ystep, y.max() + ystep
    h = max((x_max - x_min, y_max - y_min))/h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    features = np.c_[xx.ravel(), yy.ravel()]
    if precomputer is not None:
        if type(precomputer) is RBFSampler:
            features = precomputer.transform(features)
        elif precomputer is rbf_kernel:
            features = rbf_kernel(features, X)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(features)
    else:
        Z = clf.predict_proba(features)
    if len(Z.shape) > 1:
        Z = Z[:, 1]

    # Put the result into a color plot
    cm = plt.cm.RdBu_r
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    del xx, yy, x_min, x_max, y_min, y_max, Z, cm

#%% create train and test sets 
X = data.drop('label', axis=1)
Y = data['label']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=5, stratify=Y)


#%% reduce dimensionality with PCA and classify with linear discriminant analysis
#!!!!!! er wordt een voorspelling gemaakt obv training data ipv te testen met test data

x_reduced = PCA(n_components=2).fit_transform(x_train)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Two informative features, one cluster per class",
             fontsize='small')
ax.scatter(x_reduced[:, 0], x_reduced[:, 1], marker='o', c=y_train,
           s=25, edgecolor='k', cmap=plt.cm.Paired)
lda = LinearDiscriminantAnalysis()
lda = lda.fit(x_reduced, y_train)
y_pred = lda.predict(x_reduced)
colorplot(lda, ax, x_reduced[:, 0], x_reduced[:, 1])
print("Number of mislabeled points out of a total %d points : %d" % (x_reduced.shape[0], (y_train != y_pred).sum()))


#%% try out other classifiers
#!!!!!! er wordt een voorspelling gemaakt obv training data ipv te testen met test data

#   - GaussianNB
#   - LinearDiscriminantAnalysis
#   - QuadraticDiscriminantAnalysis
#   - LogisticRegression
#   - SGDClassifier
#   - KNeighborsClassifier
#   Motivate your choice. You can use the example code below to loop over both
#   the datasets and the classifiers at the same time:


clsfs = [LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),GaussianNB(),
         LogisticRegression(),SGDClassifier(),KNeighborsClassifier()]
Xs = [x_reduced]
Ys = [y_train]
clfs_fit = list()

# First make a plot without classifiers:
fig = plt.figure(figsize=(21,7*len(clsfs)))
num = 0  # Iteration number for the subplots
for X, Y in zip(Xs, Ys):
    ax = fig.add_subplot(6, 3, num + 1)
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
               s=25, edgecolor='k', cmap=plt.cm.Paired)
    num += 1

# Fit the classifiers and add them to the plot
num=0
Xt=list()
Yt=list()
for clf in clsfs:
    for X, Y in zip(Xs, Ys):
        # Fit classifier
        clf.fit(X,Y)
        y_pred=clf.predict(X)
        # Predict labels using fitted classifier

        # Make scatterplot of features
        ax = fig.add_subplot(6, 3, num + 1)
        ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
               s=25, edgecolor='k', cmap=plt.cm.Paired)
        colorplot(clf, ax, X[:,0], X[:,1])
        # Add overlay through colorplot function
        t=("Misclass: %d / %d" % ((Y!=y_pred).sum(), X.shape[0]))
        ax.set_title(t)
        num+=1

        clfs_fit.append(clf)
        Xt.append(X)
        Yt.append(Y)

#%% evaluate classifiers with metrics 
#!!!!!! er wordt een voorspelling gemaakt obv training data ipv te testen met test data

# In https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameters
# you can find the metric that you want to use:
#   - Accuracy
#   - AUC
#   - F1-score
#   - precision
#   - recall

for clf, X1, Y1 in zip(clfs_fit, Xt, Yt):
    y_pred=clf.predict(X1)

    if hasattr(clf, 'predict_proba'):
    # The first column gives the probability for class = 0, so we take
    # the second which gives the probability class = 1:
        y_score = clf.predict_proba(X1)[:, 1]
    else:
       y_score = y_pred

# The hasattr function checks whether an object, function or package has
# a certain attribute. This attribute can be a subfunction, or again an
# object or function, but also things like scalars or strings.

    auc=metrics.roc_auc_score(Y1, y_score)
    accuracy=metrics.accuracy_score(Y1, y_pred)
    F1=metrics.f1_score(Y1,y_pred)
    precision=metrics.precision_score(Y1,y_pred)
    recall=metrics.recall_score(Y1, y_pred)
# accuracy, AUC, f1score, precision, recall
    print(type(clf))
    print('Accuracy:' +str(accuracy))
    print('AUC:' +str(auc))
    print('F1:' +str(F1))
    print('precision:' +str(precision))
    print('recall:' +str(recall))