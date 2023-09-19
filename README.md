# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Dhanalakshmi
RegisterNumber:  212222040033

#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(20)
dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)
dataset

dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset

dataset.shape
(215, 10)

dataset.info()

#catgorising for further labeling
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset.dtypes

dataset.info()

dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset
dataset.info()
dataset

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()
x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])

*/
```

## Output:
![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/de5d941c-4023-486a-b111-5e4e47c9fb7c)

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/f4923b8e-3e29-4f8a-91b8-a62d75241216)

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/681f2e45-11f3-436e-b280-eaebf3af1ddc)

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/7c33b7d4-ed7a-457a-a0e0-42343e748bf1)

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/5559febb-524c-48b1-8866-c408c1bb5b6a)


![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/e6d0ce35-6e55-4eef-b11c-aeebb1356bc9)


![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/fceb1521-b73b-4ae8-a975-b34cc4a69914)


![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/01c5a594-2c55-44bb-ac56-5f8b464ddb24)



![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/c7ca28c2-5fa8-4819-a134-944b07073f3c)


![image](https://github.com/DhanalakshmiCSE/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477832/7238371a-3449-4f5a-8dff-a34978d3d703)














## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
