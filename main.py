#!/usr/bin/env python
# coding: utf-8

# # Company Churn Prediction Using Logistic Regression

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# In[3]:


df = pd.read_csv("customer_churn_dataset.csv")

# In[4]:


df.shape

# In[5]:


df.isnull().sum()

# In[6]:


df.describe()

# In[7]:


df["churn"].value_counts()

# In[8]:


sns.countplot(data=df, x="churn")
plt.title("Customer Churn Distribution")
plt.show()

# In[9]:


X = df.drop(["customer_id", "churn"], axis=1)

y = df["churn"].map({
    "No": 0,
    "Yes": 1
})

# In[10]:


numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# In[11]:


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# In[13]:


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# In[14]:


model.fit(X_train, y_train)

# In[15]:


y_pred = model.predict(X_test)

# In[17]:


accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

# In[18]:


print(classification_report(y_test, y_pred))

# In[19]:


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# In[20]:


y_probability = model.predict_proba(X_test)[:, 1]

print("ROC-AUC Score:",
      roc_auc_score(y_test, y_probability))

# In[21]:


new_customer = pd.DataFrame([{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 4,
    "phone_service": "Yes",
    "internet_service": "Fiber optic",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 95.50
}])

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Customer may leave the company")
else:
    print("Customer may stay with the company")

# In[22]:


probability = model.predict_proba(new_customer)[0][1]

print("Churn Probability:",
      probability * 100, "%")
