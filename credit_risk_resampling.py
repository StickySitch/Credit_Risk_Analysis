#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Importing dependencies
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


# # Read CSV in and Perform Basic Data Cleaning

# In[20]:


# Setting the Output column and variable columns
columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

target = ["loan_status"]


# In[21]:


# Loading in the LoanStats Data
filePath = Path('LoanStats_2019Q1.csv')
# Creating dataframe from CSV contents
df = pd.read_csv(filePath, skiprows=1)[:-2]
df = df.loc[:, columns].copy()
df.head()


# In[22]:



# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
df.count()


# In[23]:


# Dropping the null rows
df = df.dropna()
df.head()


# In[24]:


# Removing the "Issued" loan status
issuedMask = df['loan_status'] != "Issued"
df = df.loc[issuedMask]
df.head()


# In[25]:


# Converting interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%','')
df['int_rate'] = df['int_rate'].astype('float') / 100

df.head()


# In[26]:


# Converting the target column values to low_risk
x = {'Current': 'low_risk'}
df = df.replace(x)
df.head()


# In[27]:


# Converting the target column values to high_risk based on their values
x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')
df = df.replace(x)

# Checking our target columns value distribution
df.loan_status.value_counts()


# In[28]:


df.reset_index(inplace=True,drop=True)
df.head()


# In[29]:


# Converting string columns to binary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2 = df.copy()
df2['home_ownership'] = le.fit_transform(df2['home_ownership'])

df2['verification_status'] = le.fit_transform(df2['verification_status'])

df2['issue_d_Mar2019'] = le.fit_transform(df2['issue_d'])

df2['pymnt_plan'] = le.fit_transform(df2['pymnt_plan'])

df2['initial_list_status'] = le.fit_transform(df2['initial_list_status'])

df2['next_pymnt_d'] = le.fit_transform(df2['next_pymnt_d'])

df2['application_type'] = le.fit_transform(df2['application_type'])

df2['hardship_flag'] = le.fit_transform(df2['hardship_flag'])

df2['debt_settlement_flag'] = le.fit_transform(df2['debt_settlement_flag'])

df2['issue_d'] = le.fit_transform(df2['issue_d'])

df2.head()


# # Splitting Data into Training and Testing Sets

# In[30]:


# Creating variable data set
X = df2.drop(columns='loan_status', axis=1)

# Creating output data set (Target)
y = df['loan_status']
X.describe()


# In[31]:


# Checking our target columns value distribution
y.value_counts()


# In[32]:


from sklearn.model_selection import train_test_split

# Splitting data into training and testing sets
XTrain, XTest, yTrain, yTest = train_test_split(X,
                                                y,
                                                random_state=1,
                                                stratify=y
                                                )
XTrain.head()


# # Random Oversampling

# In[33]:


from imblearn.over_sampling import RandomOverSampler
# Instantiating RandomOverSampler
ros = RandomOverSampler(random_state=1)

# fitting and resampling
XResampled, yResampled = ros.fit_resample(XTrain, yTrain)


# Checking the outputs resampled value count
Counter(yResampled)


# In[34]:


# Logistic regression using random oversampled data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(XResampled, yResampled)


# In[35]:


# Calculating the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score

yPred = model.predict(XTest)
balanced_accuracy_score(yTest,yPred)


# In[36]:


# Creating confusion matrix
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(yTest,yPred),index=['Actually True', 'Actually False'], columns=['Predicted True', 'Predicted False'])


# In[37]:


# Printing the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(yTest,yPred))


# # SMOTE
# ### An oversampling technique that generates synthetic samples from the minority class

# In[38]:


from imblearn.over_sampling import SMOTE

# Using SMOTE to create our datasets (variables and target) of equal size
XResampledSMOTE, yResampledSMOTE = SMOTE(random_state=1,
                               sampling_strategy='auto').fit_resample(XTrain,yTrain)
# Checking the
Counter(yResampled)


# In[39]:


# Using logistic Regression model to make predictions
modelSMOTE = LogisticRegression(solver='lbfgs', random_state=1)
modelSMOTE.fit(XResampledSMOTE,yResampledSMOTE)


# In[40]:


# Viewing modelSMOTE balanced_accuracy_score
yPredSMOTE = modelSMOTE.predict(XTest)
balanced_accuracy_score(yTest,yPredSMOTE)


# In[41]:


# Displaying Confusion matrix
pd.DataFrame(confusion_matrix(yTest,yPredSMOTE),index=['Actually True', 'Actually False'], columns=['Predicted True', 'Predicted False'])


# In[42]:


# Displaying the imbalanced classification report
print(classification_report_imbalanced(yTest,yPredSMOTE))


# # ClusterCentroids Undersampling
# ### Method that under samples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm. This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class and using the coordinates of the N cluster centroids as the new majority samples.

# In[43]:


# Resampling the data using "ClusterCentroids" undersampling technique
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1)

XResampledCC, yResampledCC = cc.fit_resample(XTrain,yTrain)

Counter(yResampledCC)


# In[44]:


# # Using logistic Regression model to make predictions
modelCC = LogisticRegression(solver='lbfgs', random_state=1)
modelCC.fit(XResampledCC,yResampledCC)


# In[45]:


# Displaying the confusion matrix
yPredCC = modelCC.predict(XTest)
pd.DataFrame(confusion_matrix(yTest,yPredCC),index=['Actually True', 'Actually False'], columns=['Predicted True', 'Predicted False'])


# In[46]:


# Viewing modelSMOTE balanced_accuracy_score

balanced_accuracy_score(yTest,yPredCC)


# In[47]:


# displaying imbalanced classification report
print(classification_report_imbalanced(yTest,yPredCC))


# # SMOTEENN
# ### Over-sampling using SMOTE and cleaning using ENN. Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.

# In[48]:


# Using SMOTEENN to resample the data
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
XResampledENN, yResampledENN = smote_enn.fit_resample(X,y)
Counter(yResampledENN)


# In[49]:


# Fit a Logistic regression model using SMOTEENN sampling data
modelENN = LogisticRegression(solver='lbfgs', random_state=1)
modelENN.fit(XResampledENN,yResampledENN)


# In[50]:


# Displaying confusion matrix
yPredENN = modelENN.predict(XTest)
pd.DataFrame(confusion_matrix(yTest, yPredENN),index=['Actually True', 'Actually False'], columns=['Predicted True', 'Predicted False'])


# In[51]:


# Displaying Balanced Accuracy score
balanced_accuracy_score(yTest,yPredENN)


# In[52]:


# Displaying the imbalanced classification report
print(classification_report_imbalanced(yTest,yPredENN))

