#%% import packages 
import pandas as pd
import os
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import decomposition

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel
import seaborn as sns

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
print("Number of mislabeled points in the training set out of a total %d points : %d" % (x_reduced.shape[0], (y_train != y_pred).sum()))

# test met test set
pca = PCA(n_components=2)
x_train_reduced = pca.fit_transform(x_train)
x_test_reduced = pca.transform(x_test)  

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Two informative features, one cluster per class",
             fontsize='small')
ax.scatter(x_train_reduced[:, 0], x_train_reduced[:, 1], marker='o', c=y_train,
           s=25, edgecolor='k', cmap=plt.cm.Paired)
lda = LinearDiscriminantAnalysis()
lda = lda.fit(x_train_reduced, y_train)
y_pred = lda.predict(x_test_reduced)
colorplot(lda, ax, x_test_reduced[:, 0], x_test_reduced[:, 1])
print("Number of mislabeled points in the test set out of a total %d points : %d" % (x_test_reduced.shape[0], (y_test != y_pred).sum()))

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

#from the results it seems that these simple classifiers won't be enough

#%% nearest neighbour classification
k_list = [1, 3, 7]
fig = plt.figure(figsize=(24,8*len(k_list)))
num = 0

for k in k_list:
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(x_train_reduced, y_train)

    # Test the classifier on the training data and plot
    score_train = clf_knn.score(x_train_reduced, y_train)

    num += 1
    ax = fig.add_subplot(len(k_list), 2, num)
    ax.set_title(f"Training performance: accuracy {score_train}")
    colorplot(clf_knn, ax, x_train_reduced[:, 0], x_train_reduced[:, 1], h=1000)
    ax.scatter(x_train_reduced[:, 0], x_train_reduced[:, 1], marker='o', c=y_train,
               s=25, edgecolor='k', cmap=plt.cm.Paired)

    # Test the classifier on the test data and plot
    score_test = clf_knn.score(x_test_reduced, y_test)

    num += 1
    ax = fig.add_subplot(len(k_list), 2, num)
    ax.set_title(f"Test performance: accuracy {score_test}")
    colorplot(clf_knn, ax, x_test_reduced[:, 0], x_test_reduced[:, 1], h=1000)
    ax.scatter(x_test_reduced[:, 0], x_test_reduced[:, 1], marker='o', c=y_test,
               s=25, edgecolor='k', cmap=plt.cm.Paired)

#%% test multiple k's and plot performance with all features (not reduced with PCA)
train_scores = []
test_scores = []
k_list = list(range(1, 25, 2))

for k in k_list:
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(x_train, y_train)

    # Test the classifier on the training data and plot
    score_train = clf_knn.score(x_train, y_train)
    score_test = clf_knn.score(x_test, y_test)

    train_scores.append(score_train)
    test_scores.append(score_test)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.grid()
ax.plot(k_list, train_scores, 'o-', color="r",
        label="Training score")
ax.plot(k_list, test_scores, 'o-', color="g",
        label="Test score")
ax.legend(loc="best")

#%% best k for multiple splits of the data
k_list = list(range(1, 26, 2))
all_train = []
all_test = []

# Repeat the experiment 20 times, use 20 random splits in which class balance is retained
sss = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=0)

for train_index, test_index in sss.split(X, Y):
    train_scores = []
    test_scores = []

    split_X_train = X.iloc[train_index]
    split_y_train = Y.iloc[train_index]
    split_X_test = X.iloc[test_index]
    split_y_test = Y.iloc[test_index]

    for k in k_list:
        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf_knn.fit(split_X_train, split_y_train)

        # Test the classifier on the training data and plot
        score_train = clf_knn.score(split_X_train, split_y_train)
        score_test = clf_knn.score(split_X_test, split_y_test)

        train_scores.append(score_train)
        test_scores.append(score_test)

    all_train.append(train_scores)
    all_test.append(test_scores)


# Create numpy array of scores and calculate the mean and std
all_train = np.array(all_train)
all_test = np.array(all_test)

train_scores_mean = all_train.mean(axis=0)
train_scores_std = all_train.std(axis=0)

test_scores_mean = all_test.mean(axis=0)
test_scores_std = all_test.std(axis=0)

# Plot the mean scores and the std as shading
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.grid()
ax.fill_between(k_list, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
ax.fill_between(k_list, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
ax.plot(k_list, train_scores_mean, 'o-', color="r",
        label="Training score")
ax.plot(k_list, test_scores_mean, 'o-', color="g",
        label="Test score")
ax.legend(loc="best")

#%% view the optimal k with AUC instead of accuracy
k_list = list(range(1, 26, 2))
all_train = []
all_test = []

# Repeat the experiment 20 times, use 20 random splits in which class balance is retained
sss = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=0)

for train_index, test_index in sss.split(X, Y):
    train_scores = []
    test_scores = []

    split_X_train = X.iloc[train_index]
    split_y_train = Y.iloc[train_index]
    split_X_test = X.iloc[test_index]
    split_y_test = Y.iloc[test_index]

    for k in k_list:
        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf_knn.fit(split_X_train, split_y_train)

        # Test the classifier on the training data and plot
        train_proba = clf_knn.predict_proba(split_X_train)[:, 1]
        test_proba = clf_knn.predict_proba(split_X_test)[:, 1]

        score_train = metrics.roc_auc_score(split_y_train, train_proba)
        score_test = metrics.roc_auc_score(split_y_test, test_proba)


        train_scores.append(score_train)
        test_scores.append(score_test)

    all_train.append(train_scores)
    all_test.append(test_scores)


# Create numpy array of scores and calculate the mean and std
all_train = np.array(all_train)
all_test = np.array(all_test)

train_scores_mean = all_train.mean(axis=0)
train_scores_std = all_train.std(axis=0)

test_scores_mean = all_test.mean(axis=0)
test_scores_std = all_test.std(axis=0)

# Plot the mean scores and the std as shading
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.grid()
ax.fill_between(k_list, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
ax.fill_between(k_list, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
ax.plot(k_list, train_scores_mean, 'o-', color="r",
        label="Training score")
ax.plot(k_list, test_scores_mean, 'o-', color="g",
        label="Test score")
ax.legend()

#%% fitting an optimal k-NN classifier and evaluating it on the test set
cv_20fold = model_selection.StratifiedKFold(n_splits=10)
results = []
best_n_neighbors = []

# Loop over the folds
for validation_index, test_index in cv_20fold.split(X, Y):
    # Split the data properly
    X_validation = X.iloc[validation_index]
    y_validation = Y.iloc[validation_index]

    X_test = X.iloc[test_index]
    y_test = Y.iloc[test_index]

    # Create a grid search to find the optimal k using a gridsearch and 10-fold cross validation
    parameters = {"n_neighbors": list(range(1, 26, 2))}
    knn = neighbors.KNeighborsClassifier()
    cv_10fold = model_selection.StratifiedKFold(n_splits=10)
    grid_search = model_selection.GridSearchCV(knn, parameters, cv=cv_10fold, scoring='roc_auc')
    grid_search.fit(X_validation, y_validation)

    # Get resulting classifier
    clf = grid_search.best_estimator_
    print(f'Best classifier: k={clf.n_neighbors}')
    best_n_neighbors.append(clf.n_neighbors)

    # Test the classifier on the test data
    probabilities = clf.predict_proba(X_test)
    scores = probabilities[:, 1]

    # Get the auc
    auc = metrics.roc_auc_score(y_test, scores)
    results.append({
        'auc': auc,
        'k': clf.n_neighbors,
        'set': 'test'
    })

    # Test the classifier on the validation data
    probabilities_validation = clf.predict_proba(X_validation)
    scores_validation = probabilities_validation[:, 1]

    # Get the auc
    auc_validation = metrics.roc_auc_score(y_validation, scores_validation)
    results.append({
        'auc': auc_validation,
        'k': clf.n_neighbors,
        'set': 'validation'
    })

# Create results dataframe and plot it
results = pd.DataFrame(results)
seaborn.boxplot(y='auc', x='set', data=results)

optimal_n = int(np.median(best_n_neighbors))

#%% do the same thing but with scaling and PCA
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)

# Scale the data to be normal
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform a PCA
pca = decomposition.PCA(n_components=10)
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Fit kNN
knn = neighbors.KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train_pca, y_train)
score_train = knn.score(X_train_pca, y_train)
score_test = knn.score(X_test_pca, y_test)

# Print result
print(f"Training result: {score_train}")
print(f"Test result: {score_test}")

#%% Assignment E3.2 -- Support Vector Machines with kernels (ECG data)

def evaluate_model_on_test(clf, X_test, y_test, name="model"):
    y_pred = clf.predict(X_test)
    if hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        # fall back to decision_function if available
        if hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test)
        else:
            y_score = y_pred

    print(f"--- Evaluation: {name} ---")
    print(classification_report(y_test, y_pred, zero_division=1))
    auc_score = metrics.roc_auc_score(y_test, y_score)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"AUC: {auc_score:.3f} | Accuracy: {acc:.3f}")

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0','1'], yticklabels=['0','1'])
    plt.title(f'Confusion matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_test, np.asarray(y_score).ravel())
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0,1],[0,1],'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC: {name}')
        plt.legend()
    except Exception:
        pass


def run_e32_svm_kernels(X, Y, x_train, x_test, y_train, y_test):
    """Train and evaluate SVMs with common kernels on ECG data.

    - performs a small grid search per kernel to get reasonable hyperparameters
    - evaluates on the provided test split and plots PCA decision boundaries
    """
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    results = {}

    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train_scaled = scaler.transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    for kernel in kernels:
        print(f"\nRunning kernel: {kernel}")
        if kernel == 'poly':
            grid = {'C': [0.01, 0.1, 1, 10, 100], 'degree': [2, 3, 4]}
        elif kernel in ('rbf', 'sigmoid'):
            grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1.0]}
        else:
            grid = {'C': [0.01, 0.1, 1, 10, 100]}

        svc = SVC(kernel=kernel, probability=True, class_weight='balanced', random_state=42)
        gs = GridSearchCV(svc, grid, scoring='roc_auc', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
        gs.fit(X_train_scaled, y_train)
        best = gs.best_estimator_
        print('Best params:', gs.best_params_)

        # evaluate on test set
        evaluate_model_on_test(best, X_test_scaled, y_test, name=f'SVM-{kernel}')
        results[kernel] = {'best_estimator': best, 'best_params': gs.best_params_}

    # Visualize decision boundaries in 2D using PCA on scaled data (train PCA on train set)
    pca_vis = PCA(n_components=2)
    X_train_pca = pca_vis.fit_transform(X_train_scaled)
    X_test_pca = pca_vis.transform(X_test_scaled)

    fig = plt.figure(figsize=(12, 9))
    for i, kernel in enumerate(kernels, 1):
        ax = fig.add_subplot(2, 2, i)
        model = results[kernel]['best_estimator']

        # create a classifier that works in PCA space by composing scaler -> pca -> classifier
        # we'll re-fit a fresh SVC on PCA features for plotting clarity
        clf_plot = SVC(kernel=kernel, probability=True, class_weight='balanced', random_state=42, **{k: v for k, v in results[kernel]['best_params'].items() if k in ['C','gamma','degree']})
        clf_plot.fit(X_train_pca, y_train)

        ax.set_title(f'SVM {kernel} (PCA view)')
        ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolor='k', s=25)
        try:
            colorplot(clf_plot, ax, X_test_pca[:, 0], X_test_pca[:, 1])
        except Exception:
            pass

    plt.tight_layout()
    # Adding plt.show() here to display the plots when run!
    plt.show()

# Run E3.2 SVM kernels experiment
try:
    run_e32_svm_kernels(X, Y, x_train, x_test, y_train, y_test)
except Exception as e:
    print('E3.2 SVM kernels section encountered an error:', e)