# ML-EXP-12
# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Start

STEP 2: Data Loading and Exploration

STEP 3: Data Preparation

STEP 4: Train-Test Split

STEP 5: Text Vectorization

STEP 6: Model Training

STEP 7: Making Predictions

STEP 8: Evaluation

STEP 9: End 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: JAYACHITRA J
RegisterNumber:  212224040132
*/
```
```
import pandas as pd
data = pd.read_csv("spam.csv",encoding = 'Windows-1252') 
data.head()
data.isnull().sum()
x = data["v2"].values
y = data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```

## Output:

<img width="729" height="203" alt="Screenshot 2026-03-16 112124" src="https://github.com/user-attachments/assets/62babc39-a266-40a3-9451-7e15041fb731" />



<img width="241" height="133" alt="Screenshot 2026-03-16 112128" src="https://github.com/user-attachments/assets/ce488b89-d78e-4587-9bc8-9c3165a81612" />



<img width="541" height="68" alt="Screenshot 2026-03-16 112143" src="https://github.com/user-attachments/assets/2b8aba62-500c-40ce-a4a9-a44be4c15ce2" />



<img width="1207" height="115" alt="image" src="https://github.com/user-attachments/assets/fc157ff6-9d1a-4c57-a778-830aa4b99659" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
