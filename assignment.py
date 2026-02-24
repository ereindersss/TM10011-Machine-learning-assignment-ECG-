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
print(f'The number of NaN values in the entire dataframe: {raw_data.isnull().sum().sum()}')
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

#%% train a logistic regression model
lr_model = LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', random_state=4)
lr_model.fit(x_train, y_train)  
y_pred = lr_model.predict(x_test)  
probabilities = lr_model.predict_proba(x_test)

print(f"CL Report:",classification_report(y_test, y_pred, zero_division=1))

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

accuracy = sklm.accuracy_score(y_test, y_pred)
print('Accuracy:', round(accuracy,3))

f1 = f1_score(y_test, y_pred)
print("F1-Score:", f1)

recall = recall_score(y_test, y_pred)
print("Recall (Sensitivity):", recall)

confusion_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = confusion_matrix.ravel()
print("TN: {}, FP: {}, FN: {}, TP: {}\n".format(TN, FP, FN, TP))

sns.heatmap(confusion_matrix,
            annot=True,
            cmap="Blues",
            fmt='g',
            xticklabels=['0','1'],
            yticklabels=['0','1'])

plt.ylabel('Actual',fontsize=12)
plt.xlabel('Prediction',fontsize=12)
plt.show()

def plot_auc(labels, probs):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(labels.values.ravel(), probs[:,1].ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()

probabilities = lr_model.predict_proba(x_test)
plot_auc(y_test, probabilities)

#%% examine used features
features = x_train.columns.tolist()
print(features)

coefficients = lr_model.coef_[0]
coefficient_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients,
    'AbsCoefficient': np.abs(coefficients)
})

coefficient_list = coefficient_df[['Feature','Coefficient']]
coefficient_list = coefficient_list.set_index('Feature')
coefficient_list = coefficient_df.groupby('Feature')['Coefficient'].mean()
print(coefficient_list)

plt.figure(figsize=(10,6))
top20 = coefficient_list.sort_values(ascending=False).head(20)
top20.plot(kind='barh')
plt.xlabel('Mean absolute coefficient')
plt.title('Features with the highest absolute coefficients (top 20)')
plt.gca().invert_yaxis() 
plt.show()