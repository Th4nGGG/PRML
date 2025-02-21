# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:33:05 2022

@author: Admin
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:50:01 2022

@author: Admin
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 18:57:24 2022

@author: Admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

#import libraries 
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as prep


#column names
col = ["age","sex","cp", "trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]

#read data from 3 files
heart_disease_cle = pd.read_csv("processed.cleveland.data",sep = ',', names = col)
heart_disease_hun = pd.read_csv("processed.hungarian.data",sep = ',', names = col)
heart_disease_va = pd.read_csv("processed.va.data",sep = ',', names = col)

#replace '?' by NaN for pre-processing
heart_disease_hun = heart_disease_hun.replace("?", np.nan)
heart_disease_cle = heart_disease_cle.replace("?", np.nan)
heart_disease_va = heart_disease_va.replace("?", np.nan)

#count NaN values 
heart_disease_hun.isna().sum()
heart_disease_cle.isna().sum()
heart_disease_va.isna().sum()

#impute missing values for cleveland data
heart_disease_cle['ca'] = heart_disease_cle['ca'].fillna(heart_disease_cle['ca'].value_counts().index[0])
heart_disease_cle['thal'] = heart_disease_cle['thal'].fillna(heart_disease_cle['thal'].value_counts().index[0])

#Preprocess and prepare data
#delete rows with NaN
heart_disease_hun = heart_disease_hun.dropna()
heart_disease_va = heart_disease_va.dropna()

# #merge all dataframes together
heart_disease = pd.concat([heart_disease_cle,heart_disease_hun,heart_disease_va], ignore_index=True)
heart_disease['num'] = heart_disease['num'].replace([1,2,3,4],[1,1,1,1])
heart_disease = heart_disease.astype({"age": "float64","sex": "float64","cp": "float64", "trestbps": "float64","chol": "float64","fbs": "float64","restecg": "float64","thalach": "float64","exang": "float64","oldpeak": "float64","slope": "float64","ca": "float64","thal": "float64"})

# Data exploration and visualization
# Descriptive statistics
# shape
print(heart_disease.shape)
# head
print(heart_disease.head(20))
# descriptions
print(heart_disease.describe())
# class distribution
print(heart_disease.groupby('num').size())

# #Visualization
# # box and whisker plots
boxplot = heart_disease.plot(kind='box', subplots=True, layout=(4,4), 
                              sharex=False, sharey=False, 
                              fontsize = "small", figsize = (12,12))

#target variable - bar plot
target = ['0','1']
count = [164,141]
plt.bar(target, count, color ='maroon', width = 0.4)
plt.xlabel("Target class")
plt.ylabel("Number of records")
plt.title("Target variables distribution")
plt.show()

# histograms
histogram = heart_disease.hist(layout = (4,4), figsize = (12,12))

#correlation matrix and heat map
correlation = heart_disease.corr()
heatmap = sns.heatmap(correlation,vmin=-1, vmax=1, annot = True, fmt='.1g')


# Split-out train/test set
features = ["age","sex","cp", "trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
X = heart_disease[features]
Y = heart_disease["num"]
validation_size = 0.20
seed = 30
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)


#3 selected models: SVM, Logistic Regression, Naive Bayesian
#Run selected models to pick the best one

#Logistic Regression
#Scale data to fit model
X_train_scaled = prep.StandardScaler().fit_transform(X_train)
X_test_scaled = prep.StandardScaler().fit_transform(X_test)

#Parameters tunning
# **Parameters setup**
model = LogisticRegression()

param_grid = [{'C': [0.01, 0.1, 0.5, 1, 5, 10, 100], 'solver' :['newton-cg', 'lbfgs', 'liblinear'], 'class_weight':[None, 'balanced'] }]

# **Running the Grid Search on parameters and fit the trainning data**
grs_LR = GridSearchCV(model, param_grid)

# Output the best values
grs_LR.fit(X_train_scaled, Y_train)

print("Best Hyper Parameters (LR):",grs_LR.best_params_)

# Make a prediction and calculate metrics
# Now with this cross-validated model, we can predict the labels for the test data, which the model has not yet seen.
model_best_LR = grs_LR.best_estimator_
Y_pred_LR = model_best_LR.predict(X_test_scaled)

# **Evaluate the model**
print("____________Logistic Regression____________")
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred_LR))
print("Precision:",metrics.precision_score(Y_test, Y_pred_LR, average = 'weighted'))
print("Recall:",metrics.recall_score(Y_test, Y_pred_LR, average = 'weighted'))
print("F1-score:",metrics.f1_score(Y_test, Y_pred_LR, average = 'weighted'))
print(classification_report(Y_test, Y_pred_LR))
print(confusion_matrix(Y_test, Y_pred_LR))
print("\n")

#Naive Bayesian
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_NB = nb.predict(X_test)
# **Evaluate the model**
print("____________Naive Bayesian____________")
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred_NB))
print("Precision:",metrics.precision_score(Y_test, Y_pred_NB, average = 'weighted'))
print("Recall:",metrics.recall_score(Y_test, Y_pred_NB, average = 'weighted'))
print("F1-score:",metrics.f1_score(Y_test, Y_pred_NB, average = 'weighted'))
print(classification_report(Y_test, Y_pred_NB))
print(confusion_matrix(Y_test, Y_pred_NB))
print("\n")

# Support Vector Machine
# Tuning parameters
model = SVC()

# Scale data
X_train_scaled = prep.StandardScaler().fit_transform(X_train)
X_test_scaled = prep.StandardScaler().fit_transform(X_test)

# Parameters setup
param_grid = [
  {'C': [0.01, 0.1, 0.5, 1, 5, 10, 100], 'kernel': ['linear'], 'class_weight': ['balanced', None]},
  {'C': [0.01, 0.1, 0.5, 1, 5, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5, 'scale', 'auto'], 'kernel': ['rbf'], 'class_weight': ['balanced', None]},
  {'C': [0.01, 0.1, 0.5, 1, 5, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5, 'scale', 'auto'], 'kernel': ['poly'], 'class_weight': ['balanced', None], 'degree': [2, 3, 4, 5, 6, 7]},
  {'C': [0.01, 0.1, 0.5, 1, 5, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5, 'scale', 'auto'], 'kernel': ['sigmoid'], 'class_weight': ['balanced', None]}
]

# **Running the Grid Search on parameters and fit the trainning data**
grs_SVM = GridSearchCV(model, param_grid)

# Output the best parameters
grs_SVM.fit(X_train_scaled, Y_train)

print("Best Hyper Parameters (SVM):",grs_SVM.best_params_)

# Make a prediction and calculate metrics

# Now with this cross-validated model, we can predict the labels for the test data, which the model has not yet seen.
model_best_SVM = grs_SVM.best_estimator_
Y_pred_SVM = model_best_SVM.predict(X_test_scaled)

# Evaluate the model
print("____________SVM____________")
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred_SVM))
print("Precision:",metrics.precision_score(Y_test, Y_pred_SVM, average = 'weighted'))
print("Recall:",metrics.recall_score(Y_test, Y_pred_SVM, average = 'weighted'))
print("F1-score:",metrics.f1_score(Y_test, Y_pred_SVM, average = 'weighted'))
print(classification_report(Y_test, Y_pred_SVM))
print(confusion_matrix(Y_test, Y_pred_SVM))
print("\n")

# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression(C=0.01, class_weight = None, solver ="newton-cg")))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(C=0.5, class_weight=None, gamma = 0.1, kernel = 'sigmoid')))

# evaluate each model in turn
results = []
names = []
seed = 100
for name, model in models:
 	kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
 	cv_results = cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring='accuracy')
 	results.append(cv_results)
 	names.append(name)
 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Models Comparison (seed = 100)')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#confusion matrices
cm_LR = confusion_matrix(Y_test, Y_pred_LR)
cm_NB = confusion_matrix(Y_test, Y_pred_NB)
cm_SVM = confusion_matrix(Y_test, Y_pred_SVM)

#visualize
plt.figure(figsize=(12,12))

plt.subplot(131)
plt.title("Logistic Regression")
sns.heatmap(cm_LR, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('True label')
plt.ylabel('Predicted label')

plt.subplot(132)
plt.title("Naive Bayesian")
sns.heatmap(cm_NB, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('True label')
plt.ylabel('Predicted label')

plt.subplot(133)
plt.title("SVM")
sns.heatmap(cm_SVM, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('True label')
plt.ylabel('Predicted label')

#ROC curve 
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_LR)
roc = metrics.auc(fpr, tpr)

plt.figure(figsize=(12,12))
plt.title('ROC Curve - Logistic Regression')
plt.plot(fpr, tpr, 'b', label = 'AUC=%0.2f' % roc)
plt.legend(loc = 'upper left')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Best model
chosen_model = LogisticRegression(C=0.01, class_weight = None, solver ="newton-cg")
chosen_model.fit(X_train_scaled, Y_train)

#Save model for later use
print("\n ")
print('Model score (LR): ', chosen_model.score(X_test_scaled, Y_test))
joblib.dump(chosen_model, 'LR_model.sav')

#Uncomment this to load model
# loaded_model = joblib.load('LR_model.sav')
# print('Loaded model score: ', loaded_model.score(X_test_scaled, Y_test))
