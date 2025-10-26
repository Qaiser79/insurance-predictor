#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as se
import matplotlib.pyplot  as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[2]:


df = pd.read_csv("F:\Medical_Insuarance_Project\insurance.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


print(df.duplicated().sum())


# In[8]:


df=df.drop_duplicates()


# In[9]:


df['sex']=df['sex'].str.lower()


# In[10]:


df['region']=df['region'].str.lower()


# In[11]:


#outlier in charges as max value is too large than mean


# In[12]:


se.histplot(df["charges"], bins=50, kde=True)
plt.title('Distribution of Insurance Charges')
plt.show()


# In[13]:


# as we can see from histogram that the charges column is right skewed and a most charges are near lower end its a sign of outliers


# In[14]:


Q1= df['charges'].describe()['25%']
Q3= df['charges'].describe()['75%']


# In[15]:


IQR= Q3-Q1


# In[16]:


upper_bound= Q3+1.5*IQR


# In[17]:


upper_bound


# In[18]:


df_new=df[df['charges']<=upper_bound]


# In[19]:


df_new.info()


# In[20]:


df_new.describe()


# In[21]:


se.histplot(df_new["charges"], bins=50, kde=True)
plt.title('Distribution of Insurance Charges')
plt.show()


# In[22]:


se.boxplot(x="sex",y="charges", data=df_new)
plt.title('Insurance Charges by Smoker Status')
plt.show()


# In[23]:


se.boxplot(x="children", y="charges", data=df_new)
plt.title('Charges vs Number of Children')
plt.show()


# In[24]:


se.boxplot(x="region", y="charges", data=df_new)
plt.title('Charges by Region')
plt.show()


# In[25]:


se.boxplot(x="smoker", y="charges", data=df_new)
plt.title('Charges by Smoking status')
plt.show()


# In[26]:


# looking above chart we can see that smoker column has strong influence in deciding the insurance price


# In[27]:


#converting catagorical variables to number


# In[28]:


df_encoded= pd.get_dummies(df_new, drop_first=True)


# In[29]:


df_encoded.head()


# In[30]:


df_encoded= df_encoded.astype(int)


# In[31]:


df_encoded


# In[32]:


corr_mtx= df_encoded.corr()


# In[33]:


corr_mtx


# In[34]:


plt.figure(figsize=(10,8))
se.heatmap(corr_mtx, annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[35]:


features_list = ['age', 
        'bmi', 
        'children',
        'sex_male', 
        'smoker_yes']


# In[36]:


X= df_encoded[features_list]
y= df_encoded['charges']


# In[37]:


X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)


# In[38]:


model= LinearRegression()
model.fit(X_train,y_train)


# In[39]:


y_pred= model.predict(X_test)


# In[40]:


print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)


# In[41]:


residuals= y_test-y_pred


# In[42]:


plt.figure(figsize=(8,6))
se.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Charges")
plt.show()


# In[43]:


df_encoded['log_charges']= np.log(df_encoded['charges'])


# In[44]:


y= df_encoded['log_charges']


# In[45]:


X = df_encoded[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[47]:


y_pred_log = model.predict(X_test)
r2 = r2_score(y_test, y_pred_log)
mae = mean_absolute_error(y_test, y_pred_log)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_log))

print("R² Score (log):", r2)
print("MAE (log):", mae)
print("RMSE (log):", rmse)


# In[48]:


from sklearn.preprocessing import PolynomialFeatures


# In[49]:


poly= PolynomialFeatures(degree=3, include_bias=False)
X_poly= poly.fit_transform(X)


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)


# In[51]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[52]:


y_pred_poly = model.predict(X_test)

r2 = r2_score(y_test, y_pred_poly)
mae = mean_absolute_error(y_test, y_pred_poly)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print("R² Score (poly):", r2)
print("MAE (poly):", mae)
print("RMSE (poly):", rmse)


# In[53]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Best R² (RF):", grid_rf.best_score_)
print("Best Params (RF):", grid_rf.best_params_)


# In[64]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

gb = GradientBoostingRegressor(random_state=42)
grid_gb = GridSearchCV(gb, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_gb.fit(X_train, y_train)

print("Best R² (GB):", grid_gb.best_score_)
print("Best Params (GB):", grid_gb.best_params_)


# In[54]:


from sklearn.ensemble import GradientBoostingRegressor

final_model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)
final_model.fit(X_train, y_train)


# In[55]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

y_pred_final = final_model.predict(X_test)

r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

print("Final R²:", r2)
print("Final MAE:", mae)
print("Final RMSE:", rmse)


# In[56]:


from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso

pipeline = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Lasso(max_iter=10000, random_state=42)
)

param_grid = {
    'lasso__alpha': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

# Print best results
print("Best R² (Lasso):", grid.best_score_)
print("Best alpha:", grid.best_params_['lasso__alpha'])


# In[57]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso

final_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Lasso(alpha=0.01, max_iter=10000, random_state=42)
)
final_model.fit(X_train, y_train)


# In[58]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

y_pred_final = final_model.predict(X_test)

r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

print("Final R²:", r2)
print("Final MAE:", mae)
print("Final RMSE:", rmse)


# In[59]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet

model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    ElasticNet(alpha=0.01, l1_ratio=0.8, max_iter=10000, random_state=42)
)
model.fit(X_train, y_train)


# In[60]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score (ElasticNet):", r2)
print("MAE:", mae)
print("RMSE:", rmse)


# In[79]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import numpy as np
import joblib

# Define pipeline
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Lasso(alpha=0.01, max_iter=10000, random_state=42)
)

# Run cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Report results
print("Cross-validated R² scores:", scores)
print("Mean R²:", np.mean(scores))
print("Std Dev:", np.std(scores))


# In[80]:


#saveing the model for predictions
model.fit(X,y)


# In[ ]:





# In[81]:


joblib.dump(model,'insurance_cost_model.pkl')


# In[82]:


#test on new data


# In[83]:


loaded_model=joblib.load('insurance_cost_model.pkl')


# In[84]:


#prdict
new_data = pd.DataFrame([
    [25, 22.0, 0, 1, 0],
    [40, 30.5, 2, 0, 1],
    [55, 27.8, 1, 1, 0],
    [33, 35.0, 3, 1, 1],
    [60, 29.4, 0, 0, 0]
], columns=['age', 'bmi', 'children', 'sex_male', 'smoker_yes'])


# In[85]:


predictions= loaded_model.predict(new_data)


# In[86]:


print(predictions)


# In[ ]:





# In[ ]:




