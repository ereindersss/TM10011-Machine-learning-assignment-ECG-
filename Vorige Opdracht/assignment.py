# %% IMPORTEER LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from Functions import colorplot, plot_roc_curve

# %% DATA LADEN
data = pd.read_csv("ecg_data.csv", index_col=0)

# %% RANDOM FOREST CLASSIFIERS
clsfs = [RandomForestClassifier(n_estimators=1),
         RandomForestClassifier(n_estimators=5),
         RandomForestClassifier(n_estimators=200)]

fig = plt.figure(figsize=(16, 8*len(clsfs)+1))
ax = fig.add_subplot(4, 1, 1)
ax.scatter(data["0_0"], data["0_1"], marker='o', c=data["label"], s=25, edgecolor='k', cmap=plt.cm.Paired)

num=2
for clf in clsfs:
    X_train, X_test, y_train, y_test = train_test_split(data[["0_0", "0_1"]], data['label'], test_size=0.33)
    clf.fit(X_train, y_train)
    
    ax = fig.add_subplot(4, 1, num)
    ax.scatter(data["0_0"], data["0_1"], marker='o', c=data["label"], s=25, edgecolor='k', cmap=plt.cm.Paired)

    colorplot(clf, ax, data["0_0"], data["0_1"], data[["0_0", "0_1"]])
    y_pred = clf.predict(X_test)
    t = ("Misclassified: %d / %d" % ((y_test != y_pred).sum(), y_test.shape[0]))
    ax.set_title(t)
    num+=1


# %% DATA VOORBEREIDEN VOOR RIDGE EN LASSO CLASSIFIER
X = data.drop(columns=['label'])
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.33)

n_alphas = 200
alphas = np.logspace(-10, -1, n_alphas)


# %% RIDGE
coefs = []
accuracies = []
times = []
for a in alphas:
    clf = RidgeClassifier(alpha=a, fit_intercept=False)
    t0 = time()
    clf.fit(X_train, y_train)
    duration = time() - t0
    y_pred = clf.predict(X_test)
    message = ("\t Misclassified: %d / %d" % ((y_test != y_pred).sum(), y_test.shape[0]))
    print(message)

    accuracy = float((y_test != y_pred).sum()) / float(y_test.shape[0])
    times.append(duration)
    accuracies.append(accuracy)
    coefs.append(clf.coef_)

plt.figure()
ax = plt.gca()
ax.plot(alphas, np.squeeze(coefs))
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

plt.figure()
ax = plt.gca()
ax.plot(alphas, accuracies)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('accuracies')
plt.title('Performance as a function of the regularization')
plt.axis('tight')
plt.show()

plt.figure()
ax = plt.gca()
ax.plot(alphas, times)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('times (s)')
plt.title('Fitting time as a function of the regularization')
plt.axis('tight')
plt.show()


# %% LASSO
coefs = []
accuracies = []
times = []
for a in alphas:
    clf = Lasso(alpha=a, fit_intercept=False)
    t0 = time()
    clf.fit(X_train, y_train)
    duration = time() - t0
    y_pred = clf.predict(X_test)
    message = ("\t Misclassified: %d / %d" % ((y_test != y_pred).sum(), y_test.shape[0]))
    print(message)

    accuracy = float((y_test != y_pred).sum()) / float(y_test.shape[0])
    times.append(duration)
    accuracies.append(accuracy)
    coefs.append(clf.coef_)

plt.figure()
ax = plt.gca()
ax.plot(alphas, np.squeeze(coefs))
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

plt.figure()
ax = plt.gca()
ax.plot(alphas, accuracies)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('accuracies')
plt.title('Performance as a function of the regularization')
plt.axis('tight')
plt.show()

plt.figure()
ax = plt.gca()
ax.plot(alphas, times)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('times (s)')
plt.title('Fitting time as a function of the regularization')
plt.axis('tight')
plt.show()


# %% LASSO FEATURE RANGSCHIKKEN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LDA()
clf.fit(X_train_scaled, y_train)
y_score = clf.predict_proba(X_test_scaled)
plot_roc_curve(y_score, y_test)

selector = SelectFromModel(estimator=LogisticRegression(penalty='l1', solver='liblinear', C=1), threshold='mean')
selector.fit(X_train_scaled, y_train)
X_train_transformed = selector.transform(X_train_scaled)
X_test_transformed = selector.transform(X_test_scaled)
print(f"Selected {X_train_transformed.shape[1]} from {X_train_scaled.shape[1]} features.")

clf = LDA()
clf.fit(X_train_transformed, y_train)
y_score = clf.predict_proba(X_test_transformed)
plot_roc_curve(y_score, y_test)


# %% RANDOM FOREST VERSIE 2
param_distributions = {'n_estimators': [500],
                       'max_depth': randint(5, 40),
                       'min_samples_split': randint(2, 20),
                       'min_samples_leaf': randint(1, 10),
                       'max_features': ['sqrt', 'log2']}

clf = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1), param_distributions, cv=5, n_iter=20, n_jobs=-1)
clf.fit(X_train, y_train)

print('\n The best estimator, parameters and score are:')
print(f'\t {clf.best_estimator_}')
print(f'\t {clf.best_params_}')
print(f'\t {clf.best_score_}')

print(clf.cv_results_)
# %% RANDOM FOREST MAAR DAN ALLEEN MET GESELECTEERDE FEATURES
param_distributions = {'n_estimators': [500],
                       'max_depth': randint(5, 40),
                       'min_samples_split': randint(5, 20),
                       'min_samples_leaf': randint(5, 10),
                       'max_features': ['sqrt', 'log2']}

selector = SelectKBest(score_func=f_classif, k='all')  # keep all for now
X_selected_all = selector.fit_transform(X_train, y_train)
scores = selector.scores_

mask = scores >= 5

X_selected_threshold = X.iloc[:, mask]
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_selected_threshold, data['label'], test_size=0.33)

clf = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1), param_distributions, cv=5, n_iter=20, n_jobs=-1)
clf.fit(X_train_top, y_train_top)

print('\n The best estimator, parameters and score are:')
print(f'\t {clf.best_estimator_}')
print(f'\t {clf.best_params_}')
print(f'\t {clf.best_score_}')
# %%
