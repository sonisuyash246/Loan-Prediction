#!/usr/bin/env python
# coding: utf-8

# # Problem Statment

# A Company wants to automate the loan eligibility process based on customer details provided while filling online application form. The details filled by the customer are Gender, Marital Status, Education, Number of Dependents, Income of self and co applicant, Required Loan Amount, Required Loan Term, Credit History and others. The requirements are as follows: 1.Check eligibility of the Customer given the inputs described above.(Classification) 2.Identify customer segments from given data and categorize customer into one of the segments.(Clustering) 3.)If customer is not eligible for the input required amount and duration: a.)what can be amount for the given duration.(Regression) b.)if duration is less than equal to 20 years, is customer eligible for required amount for some longer duration?(Regression)
# 1.Check eligibility of the Customer given the inputs described above.(Classification)

# # Import Lib. and Filterwarnings

# In[2]:


import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings("ignore")


# # Read Dataset

# In[3]:


train = pd.read_csv("C:/Users/hp/OneDrive/Desktop/DS/project2/training_set.csv")
test =pd.read_csv("C:/Users/hp/OneDrive/Desktop/DS/project2/testing_set.csv")


# In[4]:


train.head(3)


# In[6]:


train.columns


# In[8]:


train.describe()


# In[13]:


train.shape


# In[9]:


test.head(3)


# In[10]:


test.describe()


# In[11]:


test.columns


# In[14]:


test.shape


# # MISSING DATA TREATMENT

# In[7]:


train.isna().sum()


# In[8]:


from function import catconsep
catconsep(train)


# In[9]:


from function import replacer
replacer(train)
train.isna().sum()


# # Exploratory data analysis(EDA)

# In[18]:


pvals = []
from function import ANOVA,chisq
for i in train.columns:
    if(train[i].dtypes == "object"):
        pval = chisq(train,"Loan_Status",i)
        pvals.append(pval)
    else:
        pval = ANOVA(train,"Loan_Status",i)
        pvals.append(pval)
        
t3 = pd.DataFrame([train.columns,pvals]).T
t3.columns=["Col","Pval"]
t3.sort_values(by="Pval",ascending=False)


# In[19]:


pvals


# In[20]:


from function import replacer
replacer(train)
replacer(test)


# # Define X and Y

# In[21]:


Y = train[["Loan_Status"]]
X = train.drop(labels=["Loan_ID","Loan_Status"],axis=1)


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(15,12))
plt.subplot(3,3,1)
sb.countplot(train.Married,hue=train.Loan_Status)
plt.subplot(3,3,2)
sb.countplot(train.Education,hue=train.Loan_Status)
plt.subplot(3,3,3)
sb.countplot(train.Self_Employed,hue=train.Loan_Status)
plt.subplot(3,3,4)
sb.countplot(train.Property_Area,hue=train.Loan_Status)
plt.subplot(3,3,5)
sb.countplot(train.Dependents,hue=train.Loan_Status)
plt.subplot(3,3,6)
sb.countplot(train.Gender,hue=train.Loan_Status)


# # CatConSep

# In[24]:


from function import catconsep
cat,con = catconsep(X)


# In[25]:


cat


# In[27]:


con


# # Preprocessing

# In[28]:


from function import preprocessing
Xnew= preprocessing(X)


# In[29]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = X[con]
X2 = pd.get_dummies(X[cat])
Xnew = X1.join(X2)


# In[30]:


Xtest = test.drop(labels=["Loan_ID"],axis=1)
cat_test,con_test = catconsep(Xtest)
Xtest_new =(Xtest)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1_test = Xtest[con]
X2_test = pd.get_dummies(Xtest[cat])
Xtest_new = X1_test.join(X2)


# In[32]:


Xnew


# In[33]:


Xtest


# # Train Test split

# In[34]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)


# In[109]:


tr = []
ts = []
from sklearn.tree import DecisionTreeClassifier
for i in range(2,20,1):
    
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=i)
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
import matplotlib.pyplot as plt
plt.plot(range(2,20,1),tr,c="green")
plt.plot(range(2,20,1),ts,c="red")


# In[110]:


tr = []
ts = []
from sklearn.tree import DecisionTreeClassifier
for i in range(2,70,1):
    
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=i)
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
import matplotlib.pyplot as plt
plt.plot(range(2,70,1),tr,c="green")
plt.plot(range(2,70,1),ts,c="red")


# In[111]:


tr = []
ts = []
from sklearn.tree import DecisionTreeClassifier
for i in range(2,150,1):
    
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=i)
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
import matplotlib.pyplot as plt
plt.plot(range(2,150,1),tr,c="green")
plt.plot(range(2,150,1),ts,c="red")


# In[112]:


tr = []
ts = []
from sklearn.tree import DecisionTreeClassifier
for i in range(2,150,1):
    
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=i)
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
import matplotlib.pyplot as plt
plt.plot(range(2,150,1),tr,c="green")
plt.plot(range(2,150,1),ts,c="red")


# In[113]:


tr = []
ts = []
from sklearn.tree import DecisionTreeClassifier
for i in range(2,200,1):
    
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=i)
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
import matplotlib.pyplot as plt
plt.plot(range(2,200,1),tr,c="green")
plt.plot(range(2,200,1),ts,c="red")


# In[40]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=4)

model_dtc = dtc.fit(xtrain,ytrain)
pred_tr = model_dtc.predict(xtrain)
pred_ts = model_dtc.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,pred_tr)
ts_acc = accuracy_score(ytest,pred_ts)


# In[41]:


tr_acc


# In[42]:


ts_acc


# In[43]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytrain,pred_tr))
print(confusion_matrix(ytest,pred_ts))


# adaboost

# In[60]:


tr = []
ts = []
for i in range(2,20,1):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=i)
    abc = AdaBoostClassifier(dtc,n_estimators=30)
    model = abc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
import matplotlib.pyplot as plt
plt.plot(range(2,20,1),tr,c="green")
plt.plot(range(2,20,1),ts,c="red")


# # Random Forest Classifier(RFC)

# In[46]:


tr = []
ts = []
for i in range(2,20,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,max_depth=i)
    model = rfc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
    
import matplotlib.pyplot as plt
plt.plot(range(2,20,1),tr,c="green")
plt.plot(range(2,20,1),ts,c="red")


# In[53]:


tr = []
ts = []
for i in range(2,100,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,max_depth=i)
    model = rfc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
    
import matplotlib.pyplot as plt
plt.plot(range(2,100,1),tr,c="green")
plt.plot(range(2,100,1),ts,c="red")


# In[52]:


tr = []
ts = []
for i in range(2,150,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,max_depth=i)
    model = rfc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
    
import matplotlib.pyplot as plt
plt.plot(range(2,150,1),tr,c="green")
plt.plot(range(2,150,1),ts,c="red")


# In[54]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,max_depth=5)
model = rfc.fit(xtrain,ytrain)
pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,pred_tr)
ts_acc = accuracy_score(ytest,pred_ts)


# In[55]:


tr_acc


# In[56]:


ts_acc


# In[57]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytrain,pred_tr))
print(confusion_matrix(ytest,pred_ts))


# In[61]:


tr = []
ts = []
for i in range(2,20,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,min_samples_split=i)
    abc = AdaBoostClassifier(rfc,n_estimators=30)
    model = abc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
    
import matplotlib.pyplot as plt
plt.plot(range(2,20,1),tr,c="green")
plt.plot(range(2,20,1),ts,c="red")


# In[62]:


tr = []
ts = []
for i in range(300,350,2):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,min_samples_split=i)
    abc = AdaBoostClassifier(rfc,n_estimators=30)
    model = abc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)
    
    
import matplotlib.pyplot as plt
plt.plot(range(300,350,2),tr,c="green")
plt.plot(range(300,350,2),ts,c="red")


# # Logistic Regression(LR)

# In[63]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)


# In[64]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)
pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

from sklearn.metrics import accuracy_score
print(round(accuracy_score(ytrain,pred_tr),2))
print(round(accuracy_score(ytest,pred_ts),2))


# In[65]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytrain,pred_tr))
print(confusion_matrix(ytest,pred_ts))


# # Decision tree last model

# In[67]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=5)

model_dtc = dtc.fit(xtrain,ytrain)
pred_tr = model_dtc.predict(xtrain)
pred_ts = model_dtc.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,pred_tr)
ts_acc = accuracy_score(ytest,pred_ts)

print(round(tr_acc,4))
print(round(ts_acc,4))


# # Testing Set

# In[68]:


Xtest_new


# In[69]:


pred_testset = model_dtc.predict(Xtest_new)


# In[70]:


pred_testset


# In[71]:


test


# In[72]:


R = test[["Loan_ID"]]
R["Loan_Status"]=pred_testset


# In[73]:


R


# Problems statement -2
# Identify customer segments from given data and categorize customer into one of the segments.(Clustering)

# In[74]:


train


# In[75]:


A= train[["Property_Area"]]


# In[76]:


A


# In[77]:


bnew=pd.get_dummies(A)


# In[78]:


bnew


# # Clustering

# In[79]:


from sklearn.cluster import KMeans
pm= KMeans(n_clusters=3)
model = pm.fit(bnew)


# In[80]:


model.labels_


# In[81]:


bnew['Cluster_no']=model.labels_


# In[82]:


bnew["Property_Area"]=train.Property_Area
bnew['LoanAmount']=train.LoanAmount


# In[83]:


bnew.sort_values(by="Cluster_no")


# In[85]:


k = input("Enter the Property_Area:")
cluster_no_of_input = bnew[bnew.Property_Area==k].Cluster_no.values[0]
max_loan=bnew[(bnew['Cluster_no'] == cluster_no_of_input)].LoanAmount.max()
min_loan=bnew[(bnew['Cluster_no'] == cluster_no_of_input)].LoanAmount.min()


# In[86]:


max_loan


# In[87]:


min_loan


# In[89]:


k = input("Enter the Property_Area:")
cluster_no_of_input = bnew[bnew.Property_Area==k].Cluster_no.values[0]
max_loan=bnew[(bnew['Cluster_no'] == cluster_no_of_input)].LoanAmount.max()
min_loan=bnew[(bnew['Cluster_no'] == cluster_no_of_input)].LoanAmount.min()


# In[90]:


max_loan


# In[91]:


min_loan


# In[92]:


k = input("Enter the Property_Area:")
cluster_no_of_input = bnew[bnew.Property_Area==k].Cluster_no.values[0]
max_loan=bnew[(bnew['Cluster_no'] == cluster_no_of_input)].LoanAmount.max()
min_loan=bnew[(bnew['Cluster_no'] == cluster_no_of_input)].LoanAmount.min()


# In[93]:


max_loan


# In[94]:


min_loan


# Problems Statement-3
# If customer is not eligible for the input required amount and duration: a.)what can be amount for the given duration.(Regression) b.)if duration is less than equal to 20 years, is customer eligible for required amount for some longer duration?(Regression)

# In[95]:


train


# In[96]:


R=pred=model.predict


# In[97]:


tr=["Loan_ID"]
ts=["Loan_Amount"]


# In[98]:


tr


# In[99]:


ts


# In[101]:


X1


# In[102]:


X2


# In[103]:


pred_tr


# In[104]:


pred_ts


# In[105]:


Loan_prediction=pd.DataFrame(pred_tr)


# In[106]:


Loan_prediction=pd.DataFrame(pred_ts)


# In[108]:


Loan_prediction.to_csv("C:/Users/hp/OneDrive/Desktop/DS/lp.csv")


# In[ ]:




