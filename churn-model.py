#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report
data = pd.read_csv("Telco-Customer-Churn.csv")

# data cleaning
data[data["TotalCharges"]==" "].shape
data["TotalCharges"] = data["TotalCharges"].replace(" ",np.nan)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])
data = data.dropna()

#removing unnecessary column
data = data.drop("customerID", axis=1)


#encode the target variable 'Churn'
# Yes -> 1 (customer churn) , No -> 0 (customer stayed)
data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

#Check the distribution of target variable 'Churn'
#print(data["Churn"].value_counts())
#print("\n")
# checking the cateogry of values in each column
#for col in data.columns:
   # print(col, data[col].unique())

# Encode binary category variables which have two categories like yes or no, male or female
binary_cols = ["gender","Partner","Dependents","PhoneService","PaperlessBilling"]
for col in binary_cols:
    data[col] = data[col].map({'Yes':1, 'No':0,'Male':1, 'Female': 0})

# finding multi category columns
categorical_cols = data.select_dtypes(include=['object','string']).columns

# One-hot encode all multi category categorical variables to convert them into numeric (0/1) columns.
# drop_first=True removes the first category of each variable to avoid redundancy (dummy variable trap)
# and prevent multi collinearity in models like Logistic Regression.

data = pd.get_dummies(data,columns=categorical_cols,drop_first=True)


# splitting the dataset into features (X) and target (Y)
y = data["Churn"]
x= data.drop('Churn', axis = 1)

#train and test data
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# scaling the specific columns

scaler = StandardScaler()
num_cols = ['tenure','MonthlyCharges','TotalCharges']
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

# creating logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train,y_train)
y_pred_lr = log_reg.predict(x_test)

print("logistic regression performance")
print("Accuracy: ",accuracy_score(y_test,y_pred_lr))
print("Precision",precision_score(y_test,y_pred_lr))
print("Recall",recall_score(y_test,y_pred_lr))
print("F1-score",f1_score(y_test,y_pred_lr))


# creating Decision tree model

dt = DecisionTreeClassifier(max_depth=5,random_state=42)
dt.fit(x_train,y_train)
y_pred_dt = dt.predict(x_test)

print("Decision Tree Performance")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1-score:", f1_score(y_test, y_pred_dt))