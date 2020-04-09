#!/usr/bin/env python
# coding: utf-8

# # Part 1: Regression on California Test Scores

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Caschool.csv', index_col = 0)


# In[5]:


df


# ## Q1.1 

# In[4]:


df['testscr'].hist()
plt.xlabel('Average Test Score (read.scr+math.scr)/2')
plt.ylabel('Frequency')


# In[5]:


df['compstu'].hist()
plt.xlabel('Computer per Student')
plt.ylabel('Frequency')


# In[6]:


df['expnstu'].hist()
plt.xlabel('Expenditure per Student')
plt.ylabel('Frequency')


# In[7]:


df['str'].hist()
plt.xlabel('Student Teacher Ratio')
plt.ylabel('Frequency')


# ## Q1.2 

# In[8]:


plt.scatter(df['compstu'], df['testscr'])
plt.xlabel('Computer per Student')
plt.ylabel('Average Test Score (read.scr+math.scr)/2')


# In[9]:


plt.scatter(df['expnstu'], df['testscr'])
plt.xlabel('Expenditure per Student')
plt.ylabel('Average Test Score (read.scr+math.scr)/2')


# In[10]:


plt.scatter(df['str'], df['testscr'])
plt.xlabel('Student Teacher Ratio')
plt.ylabel('Average Test Score (read.scr+math.scr)/2')


# ## Q1.3

# ### Split data in training and test set

# In[11]:


X = df.iloc[:, -7:-4]


# In[12]:


X.head()


# In[13]:


y = df.iloc[:, -8]
y


# In[14]:


from sklearn.model_selection import train_test_split

# Use train_test_split(X,y) to create four new data sets, defaults to .75/.25 split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train.head()


# ### KNN for regression

# In[15]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

kfold = KFold(n_splits=5)

#Print accuracy rounded to two digits to the right of decimal
print("Training set accuracy score: {:.2f}".format(knn.score(X_train, y_train)))
print("Testing set accuracy score: {:.2f}".format(knn.score(X_test, y_test)))
print("KFold:\n{}".format(
np.mean(cross_val_score(knn, X_train, y_train, cv=kfold))))


# ### Linear Regression (OLS)

# In[16]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set accuracy score: {:.2f}".format(lr.score(X_train, y_train)))
print("Testing set accuracy score: {:.2f}".format(lr.score(X_test, y_test)))
print("KFold:\n{}".format(np.mean(cross_val_score(lr, X_train, y_train, cv=kfold, scoring="r2"))))


# ### Ridge

# In[17]:


from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
print("KFold:\n{}".format(
np.mean(cross_val_score(ridge, X_train, y_train, cv=kfold))))


# ### Lasso

# In[18]:


from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

print("lasso.coef_: {}".format(lasso.coef_))
print("KFold:\n{}".format(
np.mean(cross_val_score(lasso, X_train, y_train, cv=kfold))))


# ### StandardScaler

# In[19]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) 


# #### KNN-StandardScaler

# In[20]:


knn.fit(X_train_scaled, y_train)

print("Training set accuracy score: {:.2f}".format(knn.score(X_train_scaled, y_train)))
print("Testing set accuracy score: {:.2f}".format(knn.score(X_test_scaled, y_test)))
print("KFold:\n{}".format(
np.mean(cross_val_score(knn, X_train_scaled, y_train, cv=kfold))))


# #### Linear Regression-StandardScaler

# In[21]:


lr = LinearRegression().fit(X_train_scaled, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set accuracy score: {:.2f}".format(lr.score(X_train_scaled, y_train)))
print("Testing set accuracy score: {:.2f}".format(lr.score(X_test_scaled, y_test)))
print("KFold:\n{}".format(np.mean(cross_val_score(lr, X_train_scaled, y_train, cv=kfold))))


# #### Ridge-StandardScaler

# In[22]:


from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train_scaled, y_train)

print("ridge.coef_: {}".format(ridge.coef_))
print("Training set accuracy score: {:.2f}".format(ridge.score(X_train_scaled, y_train)))
print("Test set accuracy score: {:.2f}".format(ridge.score(X_test_scaled, y_test)))
print("KFold:\n{}".format(
np.mean(cross_val_score(ridge, X_train_scaled, y_train, cv=kfold))))


# #### Lasso-StandardScaler

# In[23]:


from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train_scaled, y_train)

print("lasso.coef_: {}".format(lasso.coef_))
print("Training set accuracy score: {:.2f}".format(lasso.score(X_train_scaled, y_train)))
print("Test set accuracy score: {:.2f}".format(lasso.score(X_test_scaled, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
print("KFold:\n{}".format(
np.mean(cross_val_score(lasso, X_train_scaled, y_train, cv=kfold))))


# Scaling the data with the StandardScaler does help, especially for Ridge and Lasso model.

# ## Q1.4

# ### GridSearchCV

# #### KNN-GridSearchCV

# In[24]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())

param_grid = {'kneighborsregressor__n_neighbors': range(1, 10)}
grid = GridSearchCV(knn_pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.2f}".format(grid.score(X_test, y_test)))


# #### Linear Regression-GridSearchCV

# In[25]:


lr_pipe = make_pipeline(StandardScaler(), LinearRegression())
param_grid = {'linearregression__n_jobs': range(1, 10)}
grid = GridSearchCV(lr_pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.2f}".format(grid.score(X_test, y_test)))


# #### Ridge-GridSearchCV

# In[26]:


ridge_pipe = make_pipeline(StandardScaler(), Ridge())
param_grid = {'ridge__alpha': range(0, 10)}
grid = GridSearchCV(ridge_pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.2f}".format(grid.score(X_test, y_test)))


# #### Lasso-GridSearchCV

# In[27]:


lasso_pipe = make_pipeline(StandardScaler(), Lasso())
param_grid = {'lasso__alpha': range(0, 10)}
grid = GridSearchCV(lasso_pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.2f}".format(grid.score(X_test, y_test)))


# The result for KNN has significantly improved. The other models do not improve using GridSearchCV

# ## Q1.5

# The Ridge model with StandardScale and the linear regression model with StandardScale are my two best linear models exclusing KNN. Both of them use all three explanatory variables and the coefficient in front of them are almost the same. 

# ## Q1.6

# I choose the linear regression model with StandardScaler as the final model to predict new data because it has the highest accuracy score and K fold cross validation score. 

# # Part 2: Classification on red and white wine characteristics

# In[6]:


red = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';', index_col = 0)


# In[7]:


white = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';', index_col = 0)


# In[8]:


red['winetype'] = 1


# In[9]:


white['winetype'] = 0


# In[10]:


data = pd.concat([red, white], axis=0, sort=False)


# In[11]:


data


# ## Q2.1

# In[34]:


data['winetype'].hist()
plt.xlabel('Wine Type')
plt.ylabel('Frequency')


# In[35]:


data['volatile acidity'].hist()
plt.xlabel('Volatile Acidity')
plt.ylabel('Frequency')


# In[36]:


data['fixed acidity'].hist()
plt.xlabel('Fixed Aciditye')
plt.ylabel('Frequency')


# In[37]:


data['citric acid'].hist()
plt.xlabel('Fixed Aciditye')
plt.ylabel('Frequency')


# ## Q2.2

# In[38]:


X = data.loc[:, data.columns != 'winetype']


# In[39]:


X.head()


# In[40]:


y = data.iloc[:, -1]
y


# In[41]:


from sklearn.model_selection import train_test_split

# Use train_test_split(X,y) to create four new data sets, defaults to .75/.25 split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train.head()


# ### Logistic Regression

# In[42]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e90, solver='liblinear').fit(X_train, y_train)

print("logreg .coef_: {}".format(logreg .coef_))
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
predicted_vals = logreg.predict(X_test) # y_pred includes your predictions
print("logreg.predict: {}".format(predicted_vals))
print("KFold:\n{}".format(
np.mean(cross_val_score(logreg, X_train, y_train, cv=kfold))))


# ### Penalized Logistic Regression

# In[43]:


logregP = LogisticRegression(C=0.1, solver='liblinear').fit(X_train, y_train)

print("logreg .coef_: {}".format(logregP .coef_))
print("Training set score: {:.3f}".format(logregP.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logregP.score(X_test, y_test)))
predicted_vals = logregP.predict(X_test) # y_pred includes your predictions
print("logregP.predict: {}".format(predicted_vals))
print("KFold:\n{}".format(
np.mean(cross_val_score(logregP, X_train, y_train, cv=kfold))))


# ### KNN

# In[51]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

kfold = KFold(n_splits=5)

#Print accuracy rounded to two digits to the right of decimal
print("Training set accuracy score: {:.2f}".format(knn.score(X_train, y_train)))
print("Testing set accuracy score: {:.2f}".format(knn.score(X_test, y_test)))
print("KFold:\n{}".format(
np.mean(cross_val_score(knn, X_train, y_train, cv=kfold))))


# The accuracy scores and cross validation scores for each model are all very high and the KNN model performs relatively bad. 

# ### StandardScaler

# In[52]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) 


# #### Logistic Regression-StandardScaler

# In[53]:


logreg = LogisticRegression(C=1e90, solver='liblinear').fit(X_train_scaled, y_train)

print("logreg .coef_: {}".format(logreg .coef_))
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
predicted_vals = logreg.predict(X_test_scaled) # y_pred includes your predictions
print("logreg.predict: {}".format(predicted_vals))
print("KFold:\n{}".format(
np.mean(cross_val_score(logreg, X_train_scaled, y_train, cv=kfold))))


# #### Penalized Logistic Regression-StandardScaler

# In[54]:


logregP = LogisticRegression(C=0.1, solver='liblinear').fit(X_train_scaled, y_train)

print("logregP .coef_: {}".format(logregP .coef_))
print("Training set score: {:.3f}".format(logregP.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(logregP.score(X_test_scaled, y_test)))
predicted_vals = logregP.predict(X_test_scaled) # y_pred includes your predictions
print("logregP.predict: {}".format(predicted_vals))
print("KFold:\n{}".format(
np.mean(cross_val_score(logregP, X_train_scaled, y_train, cv=kfold))))


# #### KNN-StandardScaler

# In[55]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

kfold = KFold(n_splits=5)

#Print accuracy rounded to two digits to the right of decimal
print("Training set accuracy score: {:.3f}".format(knn.score(X_train_scaled, y_train)))
print("Testing set accuracy score: {:.3f}".format(knn.score(X_test_scaled, y_test)))
print("KFold:\n{}".format(
np.mean(cross_val_score(knn, X_train_scaled, y_train, cv=kfold))))


# Scaling the data with StandardScaler improved the results for all models with KNN model having the most significant increase in accuracy score. 

# ## Q2.3

# ### Logistic Regression-GridSearchCV

# In[56]:


# Create regularization penalty space
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)
# Create hyperparameter options
param_grid = dict(C=C, penalty=penalty)
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# ### KNN-GridSearchCV

# In[57]:


knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())

param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
grid = GridSearchCV(knn_pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# The result for KNN model improves but the result for logistic regression model does not. 

# ## Q2.4

# In[58]:


from sklearn.model_selection import StratifiedKFold

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
shuffling_kfold42 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
shuffling_kfold24 = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)


# In[59]:


#Logistic regression stratified k-fold
param_grid = dict(C=C, penalty=penalty)
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=stratified_kfold)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# In[60]:


#Logistic regression ‘kfold’ with shuffling, random_state=42
param_grid = dict(C=C, penalty=penalty)
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=shuffling_kfold42)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# In[61]:


#Logistic regression ‘kfold’ with shuffling, random_state=24
param_grid = dict(C=C, penalty=penalty)
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=shuffling_kfold24)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# In[62]:


#KNN stratified k-fold
param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
grid = GridSearchCV(knn_pipe, param_grid, cv=stratified_kfold)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# In[63]:


#KNN kfold’ with shuffling, random_state=42
param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
grid = GridSearchCV(knn_pipe, param_grid, cv=shuffling_kfold42)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# In[64]:


#KNN kfold’ with shuffling, random_state=24
param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
grid = GridSearchCV(knn_pipe, param_grid, cv=shuffling_kfold24)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# The parameters for models that can be tuned change when changing the cross-validation strategy in GridSearchCV from ‘stratified k-fold’ to ‘kfold’ with shuffling. The parameters for models that can be tuned also change when changing the random seed of the shuffling.

# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

X_train.head()


# In[66]:


#Logistic regression stratified k-fold
param_grid = dict(C=C, penalty=penalty)
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=stratified_kfold)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# In[67]:


#KNN stratified k-fold
param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
grid = GridSearchCV(knn_pipe, param_grid, cv=stratified_kfold)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test set accuracy score: {:.3f}".format(grid.score(X_test, y_test)))


# The parameters for models that can be tuned also change when changing the random state of the split into training and test data.

# ## Q2.5

# In[68]:


print("logreg .coef_: {}".format(logreg .coef_))
print("logregP .coef_: {}".format(logregP .coef_))


# Logistic regression shows higher weight towards volatile acidity, chlorides, pH and sulphates. The coefficients of penalized logistic regression are relatively small in magnitude. I choose the logistic regression as my final model because it has a higher accuracy score and cross validation score. 

# In[ ]:




