#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[4]:


from os import chdir
chdir("C:/Users/hp/Downloads/")


# # Readdataset

# In[26]:


import pandas as pd
train= pd.read_csv("C:/Users/hp/OneDrive/Desktop/DS/project2/training_set.csv")


# In[27]:


train.head()


# drop unwanted columns

# In[28]:


B = train.drop(labels=["Loan_ID"],axis=1)


# missing data treatment

# In[29]:


from function import replacer
replacer(B)


# Define X and Y

# In[30]:


Y = B[["Loan_Status"]]
X = B.drop(labels=["Loan_Status"],axis=1)


# In[31]:


from function import preprocessing
Xnew = preprocessing(X)


# preprocessing

# In[32]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y.Loan_Status = le.fit_transform(Y)


# In[33]:


Y


# In[34]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# CNN

# In[35]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[36]:


nn = Sequential()
nn.add(Dense(500,input_dim=(Xnew.shape[1])))
nn.add(Dense(500))
nn.add(Dropout(0.2))
nn.add(Dense(500))
nn.add(Dropout(0.2))
nn.add(Dense(1,activation="sigmoid"))


# In[37]:


nn.compile(loss="binary_crossentropy",metrics=["accuracy"])
model = nn.fit(xtrain,ytrain,validation_split=0.2,epochs=50)


# In[38]:


tr = model.history["loss"]


# In[39]:


ts = model.history["val_loss"]


# In[40]:


acc = model.history["accuracy"]
valacc = model.history["val_accuracy"]


# In[41]:


acc
valacc


# In[42]:


import matplotlib.pyplot as plt
plt.plot(tr,c="green")
plt.plot(ts,c="red")


# In[43]:


import matplotlib.pyplot as plt
plt.plot(acc,c="green")
plt.plot(valacc,c="red")


# In[44]:


tr_pred = []
for i in nn.predict(xtrain):
    if(i>0.5):
        tr_pred.append(1)
    else:
        tr_pred.append(0)

ts_pred = []
for i in nn.predict(xtest):
    if(i>0.5):
        ts_pred.append(1)
    else:
        ts_pred.append(0)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)
print("Training accuracy",tr_acc)
print("Testing accuracy",ts_acc)


# In[61]:


test =pd.read_csv("C:/Users/hp/OneDrive/Desktop/DS/project2/testing_set.csv")


# In[62]:


test = test.drop(labels=["Loan_ID"],axis=1)


# In[63]:


replacer(test)


# In[64]:


from function import preprocessing
test = preprocessing(test)


# In[65]:


test.columns


# In[66]:


nn.predict(test)


# In[67]:


N = []
for i in nn.predict(test):
    if(i[0] < 0.5):
        N.append(0)
    else:
        N.append(1)


# In[68]:


N


# In[69]:


pred = le.inverse_transform(N)


# In[70]:


pred


# In[74]:


test = pd.read_csv("C:/Users/hp/OneDrive/Desktop/DS/project2/testing_set.csv")


# In[75]:


D = test[["Loan_ID"]]


# In[76]:


D['Loan_Status_predicted'] = pred


# In[77]:


D


# In[78]:


D.to_csv("C:/Users/hp/OneDrive/Desktop/DS/lp2.csv")


# In[ ]:




