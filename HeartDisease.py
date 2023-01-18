import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as gol
import sklearn
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart_2020_cleaned-Copy1.csv.zip')

columns_to_drop = ['KidneyDisease', 'SkinCancer', 'Stroke']
df.drop(columns_to_drop, axis=1, inplace=True)
df.drop('Asthma', axis=1, inplace=True)
df.drop('PhysicalHealth', axis=1, inplace=True)
df.drop('MentalHealth', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
# Yes = 1 No=0
df['HeartDisease'] = df['HeartDisease'].replace(['Yes'], 1)
df['HeartDisease'] = df['HeartDisease'].replace(['No'], 0)
for column in df.columns:
  df['Smoking'] = LabelEncoder().fit_transform(df['Smoking'])
for column in df.columns:
  df['AlcoholDrinking'] = LabelEncoder().fit_transform(df['AlcoholDrinking'])
for column in df.columns:
  df['PhysicalActivity'] = LabelEncoder().fit_transform(df['PhysicalActivity'])
for column in df.columns:
  df['DiffWalking'] = LabelEncoder().fit_transform(df['DiffWalking'])
#Female = 0 Male = 1
for column in df.columns:
  df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
#No=0 Borderline = 1 Yes = 2
for column in df.columns:
  df['Diabetic'] = LabelEncoder().fit_transform(df['Diabetic'])
# 0 = excellent 1 = fair 2= good 3=poor 4= v good
for column in df.columns:
  df['GenHealth'] = LabelEncoder().fit_transform(df['GenHealth'])
df['AgeCategory'] = df['AgeCategory'].replace(['18-24'], 0)
df['AgeCategory'] = df['AgeCategory'].replace(['25-29'], 1)
df['AgeCategory'] = df['AgeCategory'].replace(['30-34'], 2)
df['AgeCategory'] = df['AgeCategory'].replace(['35-39'], 3)
df['AgeCategory'] = df['AgeCategory'].replace(['40-44'], 4)
df['AgeCategory'] = df['AgeCategory'].replace(['45-49'], 5)
df['AgeCategory'] = df['AgeCategory'].replace(['50-54'], 6)
df['AgeCategory'] = df['AgeCategory'].replace(['55-59'], 7)
df['AgeCategory'] = df['AgeCategory'].replace(['60-64'], 8)
df['AgeCategory'] = df['AgeCategory'].replace(['65-69'], 9)
df['AgeCategory'] = df['AgeCategory'].replace(['70-74'], 10)
df['AgeCategory'] = df['AgeCategory'].replace(['75-79'], 11)
df['AgeCategory'] = df['AgeCategory'].replace(['80 or older'], 12)
df['Race'] = df['Race'].replace(['White'], 5)
df['Race'] = df['Race'].replace(['Other'], 4)
df['Race'] = df['Race'].replace(['Hispanic'], 3)
df['Race'] = df['Race'].replace(['Black'], 2)
df['Race'] = df['Race'].replace(['Asian'], 1)
df['Race'] = df['Race'].replace(['American Indian/Alaskan Native'], 0)

df['BMI'] = df['BMI'].astype(int)
df['SleepTime'] = df['SleepTime'].astype(int)

target = df["HeartDisease"]
input_columns = df.drop(columns=["HeartDisease"])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_columns, target, test_size=0.61)

# Training data percentage
# x_train.shape[0] / df.shape[0]

from sklearn.neighbors import KNeighborsClassifier as KNN
df_KNN = KNN(n_neighbors=7)

# Fit data
df_KNN.fit(x_train, y_train)
# Predict data
y_hat = df_KNN.predict(x_test)
y_hat

# Accuracy / KNN
comparisons = np.array(y_hat == y_test)
comparisons.mean()

# val = [28, 1, 0, 1, 0, 11, 2, 0, 0, 1, 12]
# xdata = np.array([val])
# pred = #int(df_KNN.predict(xdata))
# [0.57142857 0.42857143]

# print(pred)
import bz2

pickle.dump(df_KNN, open("model.pkl", "wb"))
model = pickle.load(open('model.pkl','rb'))
#sfile = bz2.BZ2File('smallerfile.pkl', 'wb')
#pickle.dump(df_KNN, sfile)