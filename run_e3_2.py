#%% import packages 
import pandas as pd
import os
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn import datasets as ds
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sklearn.metrics as sklm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel

#%% load data 
with zipfile.ZipFile("ecg_data.zip","r") as zip_ref:
    zip_ref.extractall("ecg_data")

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'ecg_data/ecg_data.csv'), index_col=0)
    return data

#%% examine the data
raw_data = load_data()

print(f'The number of samples: {len(raw_data.index)}')
print(f'The number of columns: {len(raw_data.columns)}')

print(f'The number of NaN values in the entire dataframe: {raw_data.isnull().sum().sum()}')
print(f'The number of samples with label 0: {len(raw_data[raw_data["label"] == 0])}')
print(f'The number of samples with label 1: {len(raw_data[raw_data["label"] == 1])}')
print(f'The percentage of samples with label 0 is thus {len(raw_data[raw_data["label"] == 0])/len(raw_data.index)*100:.2f}%', 
      f'and the percentage with label 1 {len(raw_data[raw_data["label"] == 1])/len(raw_data.index)*100:.2f}%')

data = raw_data


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
