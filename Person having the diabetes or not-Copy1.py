#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")


# In[3]:


df


# In[4]:


#Dataset is all about whether the person having diabitics or not


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


from pandas_profiling import ProfileReport


# In[6]:


ProfileReport(df)


# In[7]:


df['BMI']=df['BMI'].replace(0,df['BMI'].mean())


# In[8]:


df.columns


# In[9]:


df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())


# In[10]:


df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())


# In[11]:


df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())


# In[12]:


df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())


# In[ ]:





# In[13]:


ProfileReport(df)


# In[14]:


fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df,ax=ax)


# In[15]:


q=df['Insulin'].quantile(0.98)
df_new=df[df['Insulin']<q]


# In[16]:


df_new


# In[17]:


fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df_new,ax=ax)


# In[ ]:





# In[18]:


df_new


# In[19]:


fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df_new,ax=ax)


# In[20]:


q=df['Pregnancies'].quantile(0.98)
df_new=df[df['Pregnancies']<q]

q=df['BMI'].quantile(0.99)
df_new=df[df['BMI']<q]

q=df['SkinThickness'].quantile(0.99)
df_new=df[df['SkinThickness']<q]

q=df['Insulin'].quantile(0.99)
df_new=df[df['Insulin']<q]

q=df['DiabetesPedigreeFunction'].quantile(0.99)
df_new=df[df['DiabetesPedigreeFunction']<q]


# In[21]:


df.columns


# In[22]:


fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df_new,ax=ax)


# In[23]:


y=df_new['Outcome']
y


# In[24]:


x=df_new.drop(columns=['Outcome'])
x


# In[25]:


df


# In[26]:


scaler=StandardScaler()
ProfileReport(pd.DataFrame(scaler.fit_transform(df_new)))
x_scaled=scaler.fit_transform(df_new)


# In[27]:


df_scaler=pd.DataFrame(scaler.fit_transform(df_new))
fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df_scaler,ax=ax)


# In[28]:


x_scaled


# In[29]:


arr=scaler.fit_transform(x)
print(arr.shape[1])
print(arr)


# # check the multi-collinearity using variance inflation factor

# In[30]:


def vif_score(x):
    scaler=StandardScaler()
    arr=scaler.fit_transform(x)
    return pd.DataFrame([[x.columns[i],variance_inflation_factor(arr,i)]for i in range(arr.shape[1])],columns=['Feature','VIF_score'])
    


# In[31]:


vif_score(x)


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=1234)


# In[33]:


logr=LogisticRegression()


# In[34]:


logr.fit(x_train,y_train)


# In[35]:


logr.predict([x_test[0]])


# In[36]:


logr.predict_proba([x_test[0]])


# In[37]:


logr.predict_log_proba([x_test[0]])


# In[38]:


#now change the solver to liblinear


# In[39]:


logr_lib_linear=LogisticRegression(verbose=1,solver='liblinear')


# In[40]:


logr_lib_linear.fit(x_train,y_train)


# In[41]:


y_pred_lib_linear=logr_lib_linear.predict(x_test)
y_pred_lib_linear


# In[42]:


y_pred=logr.predict(x_test)
y_pred


# In[43]:


confusion_matrix(y_test,y_pred)


# In[44]:


confusion_matrix(y_test,y_pred_lib_linear)


# In[45]:


auc=roc_auc_score(y_test,y_pred)
auc


# In[46]:


auc_lib_linear=roc_auc_score(y_test,y_pred_lib_linear)
auc_lib_linear


# In[47]:


fpr,tpr,thresholds=roc_curve(y_test,y_pred,)


# In[48]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[50]:


logr.score(x_test,y_test)


# In[52]:


logr.score(x_train,y_train)

