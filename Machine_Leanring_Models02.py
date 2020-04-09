#!/usr/bin/env python
# coding: utf-8

# # Machine Leanring Models

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Supress Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

pd.options.display.max_rows = 10


# ## 1. Import the spam dataset and print the first six rows.

# In[9]:


data = pd.read_csv('/Users/xianqu/Desktop/QMSS5073ML/Midterm/spam_dataset.csv')


# In[10]:


data.head(6)


# ## 2. Read through the documentation of the original dataset. The dependent variable is "spam" where one indicates that an email is spam and zero otherwise. Which three variables in the dataset do you think will be important predictors in a model of spam? Why?

# **Answer:** I think 'word_freq_credit:', 'word_freq_george:' and 'capital_run_length_average:' will be important predictors in the model of spam. Because the collection of non-spam e-mails in this dataset came from filed work and personal e-mails of one of the creators, and hence the word 'george' and the area code '650' are indicators of non-spam which should have a negative relationship with the dependent variable. For most of the ads, they put plenty of capital letters to draw people's attention to the products or services they promote so I consider 'capital_run_length_average:' to be positively correlated with variable 'spam'. Moreover, there as many spam emails nowadays regarding credit cards or loans so I choose 'word_freq_credit:' as another predictor for spam varaible. 

# ## 3. Visualize the univariate distribution of each of the variables in the previous question.

# #### spam

# In[11]:


data['spam'].hist()


# #### word_freq_credit

# In[12]:


data['word_freq_credit:'].hist()


# In[13]:


#Excluding values at zero to get a better sense at the shape of the distribution
plt.hist(data['word_freq_credit:'], bins= np.array(np.arange(1, max(data['word_freq_credit:']), 1)))
plt.xticks(np.arange(1, max(data['word_freq_credit:']), 1))
plt.show()


# #### word_freq_george

# In[14]:


data['word_freq_george:'].hist()


# In[15]:


#Excluding values at zero to get a better sense at the shape of the distribution
plt.hist(data['word_freq_george:'], bins=np.array(np.arange(1, max(data['word_freq_george:']), 1)))
plt.xticks(np.arange(1, max(data['word_freq_george:']), 2))
plt.show()


# #### capital_run_length_average

# In[16]:


data['capital_run_length_average:'].hist()


# In[17]:


#Excluding extremlly large values to get a better sense at the shape of the distribution
plt.hist(data['capital_run_length_average:'], bins=[0, 3, 6, 9, 12, 15, 18])
plt.xlim(0, 20)
plt.xticks(np.arange(0, 20, 3))
plt.show()


# ## 4. Name each of the supervised learning models that we have learned thus far that are used to predict dependent variables like "spam".

# **Answer:** K Nearest Neighbours classification model, (Penalized)Logistic Regression model, classification tree, Support Vector Machines, Decision Tree model, Bagging model, Random Forest model and Boosting model are the the supervised learning models that we have learned thus far that are used to predict catagorical dependent variables like "spam". 

# ## 5. Describe the importance of training and test data. Why do we separate data into these subsets?

# **Answer:** Training and test data are important for the model generalization. We need a model that performs well on dataset that it has never seen (test data). Training data is used to adjust the parameters of the model. The aim is to reduce bias or your predictions (i.e. to fit the data). Test data is used to provide an unbiased evaluation of a final model. The purpose of splitting data into trainging and test datasets is to avoid overfitting: ignoring unnecessary noise and only optimizing the training dataset accuracy. 

# ## 6. What is k-fold cross validation and what do we use it for?

# **Answer:** K-fold cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times, with each of the k subsamples used exactly once as the validation data. The k results can then be averaged to produce a single estimation.

# ## 7. How is k-fold cross validation different from stratified k-fold cross validation?

# **Answer:** Cross Validation: Splits the data into k "random" folds.
# 
# Stratified Cross Valiadtion: Splits the data into k folds, making sure the folds are selected so that the mean response value is approximately equal in all the folds. In the case of a dichotomous classification, this means that each fold contains roughly the same proportions of the two types of class labels.
# 
# Stratification seeks to ensure that each fold is representative of all strata of the data. Generally this is done in a supervised way for classification and aims to ensure each class is (approximately) equally represented across each test fold (which are of course combined in a complementary way to form training folds).
# 
# The intuition behind this relates to the bias of most classification algorithms. They tend to weight each instance equally which means overrepresented classes get too much weight (e.g. optimizing F-measure, Accuracy or a complementary form of error). 

# ## 8a. Choose one model from question four. Split the data into training and test subsets. 

# In[18]:


#Split data into training and test set
from sklearn.model_selection import train_test_split

y = data['spam']

X = data.loc[:, ['word_freq_credit:', 'word_freq_george:', 'capital_run_length_average:']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 

X.head()


# In[19]:


#Scale the training and testing set with standard scaler for logistic regression, KNN, SVM models. 
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## 8b. Build a model with the three variables in the dataset. Describe why you chose any particular parameters for your model. Run the model. 

# ### KNN model

# In[31]:


# Parameter selection
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())

knn_param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid).fit(X_train_scaled, y_train)

print("Best Parameter: {}".format(knn_grid.best_params_))


# **Answer:** I choose the parameter for KNN model n_neighbors to be 8, i.e. k = 8,  according to GridSearchCV.

# ## 8c. Evaluate prediction error in two ways: A) On test data directly and B) using k-fold cross-validation.

# In[32]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

print("KNN CLASSIFER (SCALED DATA)")

print("Training set score: {:.2f}".format(knn_grid.score(X_train_scaled, y_train)))
# Evaluate prediction error on test data directly
print("Test set Score: {:.2f}".format(knn_grid.score(X_test_scaled, y_test)))
# Kfold Cross Validation
kfold = KFold(n_splits=5)
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(knn_grid, X_train_scaled, y_train, cv=kfold))))


# ## 9a. Choose a second model from question four. Using the same three variables in the dataset that you think will be good predictors of "spam". Describe why you chose any particular parameters for your model. Run the model.

# ### Logistic regression

# In[62]:


from sklearn.linear_model import LogisticRegression
# Parameter selection
logreg_pipe = make_pipeline(StandardScaler(), LogisticRegression())

logreg_param_grid = {'logisticregression__C': np.linspace(1, 100, 100)}
logreg_grid = GridSearchCV(logreg_pipe, logreg_param_grid).fit(X_train_scaled, y_train)

print("Best Parameter: {}".format(logreg_grid.best_params_))


# **Answer:** I choose the parameter for logistic regression model C to be 41, according to GridSearchCV.

# ## 9b. Evaluate prediction error in two ways: A) On test data directly and B) using k-fold cross-validation. 

# In[65]:


print("LOGISTIC REGRESSION (SCALED DATA)")
print("Training set score: {:.2f}".format(logreg_grid.score(X_train_scaled, y_train)))

# Evaluate prediction error on test data directly
print("Test set Score: {:.2f}".format(logreg_grid.score(X_test_scaled, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(logreg_grid, X_train_scaled, y_train, cv=kfold))))


# ## 9c. Did this model predict test data better than your previous model?

# In[66]:


print("KNN CLASSIFIER (SCALED DATA)")
print("Test set Score: {:.2f}".format(knn_grid.score(X_test_scaled, y_test)))
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(knn_grid, X_train_scaled, y_train, cv=kfold))))
print("")
print("LOGISTIC REGRESSION (SCALED DATA)")
print("Test set Score: {:.2f}".format(logreg_grid.score(X_test_scaled, y_test)))
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(logreg_grid, X_train_scaled, y_train, cv=kfold))))


# **Answer:** The KNN classifier model predict test data better than logistic regression model based on test data score and k-fold cross validation.

# ## 10a. Choose a third model from question four. Using the same three variables in the dataset that you think will be good predictors of "spam". Describe why you chose any particular parameters for your model. Run the model.

# ### SVM

# In[64]:


#C-Support Vector Classification
from sklearn.svm import SVC
from sklearn.decomposition import PCA

#use PCA to extract 3 fundamental components to feed into SVM classifier.
#because we have 3 features
pca = PCA(svd_solver='randomized', n_components=3, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')

# Parameter selection
svc_pipe = make_pipeline(pca, svc)

svc_param_grid = {'svc__C': [1,5,10,50],
                 'svc__gamma': [0.0001,0.0005,0.001,0.005]}
svc_grid = GridSearchCV(svc_pipe, svc_param_grid).fit(X_train_scaled, y_train)

print("Best Parameter: {}".format(svc_grid.best_params_))


# **Answer:** I choose the parameters for SVC model C to be 41 and gamma to be 0.005, according to GridSearchCV.

# ## 10b. Evaluate prediction error in two ways: A) On test data directly and B) using k-fold cross-validation. 

# In[69]:


print("Support Vector Classifier (SCALED DATA)")
print("Training set score: {:.2f}".format(svc_grid.score(X_train_scaled, y_train)))

# Evaluate prediction error on test data directly
print("Test set Score: {:.2f}".format(svc_grid.score(X_test_scaled, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(svc_grid, X_train_scaled, y_train, cv=kfold))))


# ## 10c. Did this model predict test data better than your previous models?

# In[70]:


print("KNN CLASSIFIER (SCALED DATA)")
print("Test set Score: {:.2f}".format(knn_grid.score(X_test_scaled, y_test)))
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(knn_grid, X_train_scaled, y_train, cv=kfold))))
print("")
print("LOGISTIC REGRESSION (SCALED DATA)")
print("Test set Score: {:.2f}".format(logreg_grid.score(X_test_scaled, y_test)))
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(logreg_grid, X_train_scaled, y_train, cv=kfold))))
print("")
print("Support Vector Classifier (SCALED DATA)")
print("Test set Score: {:.2f}".format(svc_grid.score(X_test_scaled, y_test)))
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(svc_grid, X_train_scaled, y_train, cv=kfold))))


# **Answer:** SVC model did not perform better than KNN nor logistic regression model. KNN model is still the model that performs the best according to test score and k fold cross validation score. 

# ## 11a. Choose a fourth model from question four. Using the same three variables in the dataset that you think will be good predictors of "spam". Describe why you chose any particular parameters for your model. Run the model.

# ## Random Forest

# In[71]:


# Parameter selection
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc_param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

rfc_grid = GridSearchCV(rfc, rfc_param_grid).fit(X_train, y_train)

print("Best Parameter: {}".format(rfc_grid.best_params_))


# **Answer:** I choose the parameters for random forest model criterion to be gini, max_depth to be 8, max_features to be log2 and n_estimators to be 500, according to GridSearchCV.

# ## 11b. Evaluate prediction error in two ways: A) On test data directly and B) using k-fold cross-validation. 

# In[76]:


rfc_grid_best =RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini').fit(X_train, y_train)


# In[78]:


print("RANDOM FOREST CLASSIFIER")
print("Training set score: {:.2f}".format(rfc_grid_best.score(X_train, y_train)))

# Evaluate prediction error on test data directly
print("Test set Score: {:.2f}".format(rfc_grid_best.score(X_test, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(rfc_grid_best, X_train, y_train, cv=kfold))))


# ## 11c. Did this model predict test data better than your previous models?

# In[81]:


print("KNN CLASSIFIER (SCALED DATA)")
print("Test set Score: {:.2f}".format(knn_grid.score(X_test_scaled, y_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(knn_grid, X_train_scaled, y_train, cv=kfold))))
print("")
print("LOGISTIC REGRESSION (SCALED DATA)")
print("Test set Score: {:.2f}".format(logreg_grid.score(X_test_scaled, y_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(logreg_grid, X_train_scaled, y_train, cv=kfold))))
print("")
print("Support Vector Classifier (SCALED DATA)")
print("Test set Score: {:.2f}".format(svc_grid.score(X_test_scaled, y_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(svc_grid, X_train_scaled, y_train, cv=kfold))))
print("")
print("RANDOM FOREST CLASSIFIER")
print("Test set Score: {:.2f}".format(rfc_grid_best.score(X_test, y_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(rfc_grid_best, X_train, y_train, cv=kfold))))


# **Answer:** The random forest classification model predicted test data better than my previous models: higher test set score and cross validation score.

# ## 12a. Now rerun your best model from questions 8 through 11, but this time add three new variables to the model that you think will increase prediction accuracy. 

# In[83]:


data.columns


# In[84]:


# Add three new variables: 'word_freq_our:', 'word_freq_free:', 'word_freq_business:'
X2 = data.loc[:, ['word_freq_credit:', 'word_freq_george:', 'capital_run_length_average:', 'word_freq_our:', 'word_freq_free:', 'word_freq_business:']]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, random_state=42) 

X2.head()


# In[85]:


# Rerun Random Forest Classification model
rfc_grid2 = GridSearchCV(rfc, rfc_param_grid).fit(X2_train, y2_train)

print("Best Parameter: {}".format(rfc_grid2.best_params_))


# In[88]:


rfc_grid_best2 =RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=8, criterion='entropy').fit(X2_train, y2_train)


# ## 12b. Did this model predict test data better than your previous models?

# In[89]:


print("RANDOM FOREST CLASSIFIER")
print("Test set Score: {:.2f}".format(rfc_grid_best.score(X_test, y_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(rfc_grid_best, X_train, y_train, cv=kfold))))
print("")
print("RANDOM FOREST CLASSIFIER 2")
print("Test set Score: {:.2f}".format(rfc_grid_best2.score(X2_test, y2_test)))
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(rfc_grid_best2, X2_train, y2_train, cv=kfold))))


# **Answer:** This model predicted test data better than my previous random forest model. 

# ## 13. Rerun all your other models with this final set of six variables, evaluate prediction error, and choose a final model. Why did you select this model among all of the models that you ran?

# In[91]:


# Scale the new training and test sets
scaler2 = preprocessing.StandardScaler().fit(X2_train)
X2_train_scaled = scaler2.transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)


# In[92]:


#KNN
knn_grid2 = GridSearchCV(knn_pipe, knn_param_grid).fit(X2_train_scaled, y2_train)

print("Best Parameter: {}".format(knn_grid2.best_params_))


# In[93]:


#Logistic Regression
logreg_grid2 = GridSearchCV(logreg_pipe, logreg_param_grid).fit(X2_train_scaled, y2_train)

print("Best Parameter: {}".format(logreg_grid2.best_params_))


# In[94]:


#SVC
svc_grid2 = GridSearchCV(svc_pipe, svc_param_grid).fit(X2_train_scaled, y2_train)

print("Best Parameter: {}".format(svc_grid2.best_params_))


# In[96]:


print("KNN CLASSIFIER (SCALED DATA) 2")
print("Test set Score: {:.2f}".format(knn_grid2.score(X2_test_scaled, y2_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(knn_grid, X2_train_scaled, y2_train, cv=kfold))))
print("")
print("LOGISTIC REGRESSION (SCALED DATA) 2")
print("Test set Score: {:.2f}".format(logreg_grid2.score(X2_test_scaled, y2_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(logreg_grid, X2_train_scaled, y2_train, cv=kfold))))
print("")
print("Support Vector Classifier (SCALED DATA) 2")
print("Test set Score: {:.2f}".format(svc_grid2.score(X2_test_scaled, y2_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(svc_grid, X2_train_scaled, y2_train, cv=kfold))))
print("")
print("RANDOM FOREST CLASSIFIER 2")
print("Test set Score: {:.2f}".format(rfc_grid_best2.score(X2_test, y2_test)))
print("Mean Cross Validation, KFold: {:.3f}".format(np.mean(cross_val_score(rfc_grid_best, X2_train, y2_train, cv=kfold))))


# **Answer:** I would use Random Forest Classification model with six regressors as the final model. Because it has the highest accuracy score and cross validation score. 

# ## 14. What variable that currently is not in your model, if included, would be likely to increase your final model's predictive power? For this answer try to speculate about a variable outside the variables available in the data that would improve you model.

# **Answer:** We could also add variables such as ***mail_client*** which lists the email clients the sender used. That is, we assign each mail client a unique number e.g Outlook 2013: 1; Thunderbird:2 and so on. The frequency of the phrase/word ***Urgent***, ***Click here***, ***Casino***, ***Bonus*** etc. could also potentially increase the final model's predictive power. 

# ## 15. List each model we have focused on in class thus far that you could use to evaluate data with a continuous dependent variable.

# **Answer:** For continuous dependent variable we could use:
#     1. Linear Regression (OLS) Model;
#     2. Ridge Model;
#     3. Lasso Model;
#     4. KNN Regressor Model;
#     5a. Decision Tree Regression Model;
#     5b. Bagging Regression Model;
#     6. Random Forest Regression Model
