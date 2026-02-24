#%% import packages 
import pandas as pd
import os
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
import sklearn.metrics as sklm
from sklearn.ensemble import RandomForestClassifier
#from MLstatkit.stats import Delong_test
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import plot_tree
#from mlxtend.evaluate import mcnemar_table, mcnemar
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report

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

#print(raw_data.head())
print(f'The number of NaN values in the entire dataframe is equal to: {raw_data.isnull().sum().sum()}')
print(f'The number of samples with label 0: {len(raw_data[raw_data["label"] == 0])}')
print(f'The number of samples with label 1: {len(raw_data[raw_data["label"] == 1])}')
print(f'The percentage of samples with label 0 is thus {len(raw_data[raw_data["label"] == 0])/len(raw_data.index)*100:.2f}%', 
      f'and the percentage with label 1 {len(raw_data[raw_data["label"] == 1])/len(raw_data.index)*100:.2f}%')

# print(raw_data.groupby('label').count())
# print(raw_data.groupby('label').mean())
# print(raw_data.groupby('label').var())
# print(raw_data.groupby('label').std())

data = raw_data

#%% split data in train and test set
X = data.drop('label', axis=1)
Y = data['label']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=5, stratify=Y)

#%% principal component analysis (PCA)
categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = x_train.select_dtypes(include=['number']).columns
x_train_encoded = pd.get_dummies(x_train, columns=categorical_cols)

x_pca = StandardScaler().fit_transform(x_train_encoded)
pca_ = PCA(n_components=2)
principalComponents = pca_.fit_transform(x_pca)

principal_df = pd.DataFrame(data = principalComponents
             , columns = ['Principal Component 1', 'Principal Component 2'])
principal_df['label'] = y_train

plt.figure(figsize=(10,6))
plt.title("PCA of the training data")

sns.scatterplot(
    x="Principal Component 1", y="Principal Component 2",
    hue="label",
    palette={True: "blue", False: "green"},
    data=principal_df,
    legend="full",
    alpha=0.8
)