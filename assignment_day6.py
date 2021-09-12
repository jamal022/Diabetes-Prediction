#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import pickle 
from sklearn.metrics import accuracy_score ,confusion_matrix
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from os.path import isfile


# In[2]:


data = pd.read_csv('diabetes.csv')
data.head()


# In[3]:


data.dtypes


# In[4]:


X = data.drop(columns='Outcome')
y = data['Outcome']


# In[5]:


model = GaussianNB()
file_name = 'diabetes_model.pickle'
top_acc = 0
if not isfile(file_name):
    for i in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model.fit(X_train, y_train)        

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print('Current Accuracy : ', acc)
        if acc > top_acc:
            top_acc = acc
            with open(file_name, 'wb') as file:
                pickle.dump(model, file)
            
else:
    with open(file_name, 'rb') as file:
            model = pickle.load(file)    


# In[6]:


print('''
                                    Diabetes Classifier
                                ---------------------------
                    Please enter the characteristics of the patient you want to predict.
                                        
                                            NOTICE:
                    Please make sure that you've read the data description clearly before you enter any values.
                    
                -------------------------------------------------------------------------------------------------
                    Pregnancies: Number of times pregnant
                    Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
                    BloodPressure: Diastolic blood pressure (mm Hg)
                    SkinThickness: Triceps skin fold thickness (mm)
                    Insulin: 2-Hour serum insulin (mu U/ml)
                    BMI: Body mass index (weight in kg/(height in m)^2)
                    DiabetesPedigreeFunction: Diabetes pedigree function
                    Age: Age (years)
                    Outcome: Class variable (0 or 1)
''')


pregnancies = int(input("Enter number of times pregnant:"))
glucose = int(input("Enter concentration of Glucose:"))
blood_pressure = int(input("Enter Blood Pressure          :"))
skin_thickness = int(input("Enter Skin Thickness          :"))
insulin = int(input("Enter Insulin                 :"))
bmi = float(input("Enter BMI (Body Mass Index    :"))
diabetes_pedigree_function = float(input("Enter Diabetes Pedigree value :"))
age = int(input("Enter age of the patient      :"))

features_of_patient = np.array([pregnancies, glucose, blood_pressure, skin_thickness , insulin , bmi , diabetes_pedigree_function , age])
predictions = model.predict([features_of_patient])
print('------------------------')
if predictions == 1 :
    print("Unfortunately this patient has diabetes.") 
else:
    print("Wow! A clean bill of health") 


# In[ ]:





# In[ ]:




