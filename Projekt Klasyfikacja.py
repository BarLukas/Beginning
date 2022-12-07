#Importuje potrzebne biblioteki.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
#import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


#Wczytuję bazy danych i łączę je ze sobą.

df1 = pd.read_csv('C:/Users/lukas/df1.csv')
print('df1 shape:\n', df1.shape)
print('df1 table:\n', df1)

df2 = pd.read_csv('C:/Users/lukas/df2.csv')
print('df2 shape:\n', df2.shape)
print('df2 table:\n', df2)

df = pd.merge(df1, df2)
print('df(marged) table:\n',df)

#Sprawdzenie typu danych, wartości powtarzających się, czy brakujących danych.

print('df info:\n', df.info())

print('unique values:\n')
for i in df.columns:
    print(f"{i} ma {df[i].nunique()}")

df = df.dropna(how='any',axis=0)
print('sum of null:\n', df.isnull().sum())

#Dalsze sprawdzenie sensu i poprawności danych.

print('describe df:\n',df.describe(include="all"))

print('Age:\n', df['Age'].value_counts())

#Widać, że wiek typu 8703 lat jest niemożliwy, można założyć, że maksymalny wiek pracownika do 70lat.

median = df.loc[df['Age']<70, 'Age'].median()
df.loc[df.Age > 70, 'Age'] = np.nan
df.fillna(median, inplace=True)

#Dystans od domu o wielkości 999590 też jest niemożliwy, maksymalna odległość między dwoma oddalonymi punktami na ziemi wynosi 20000km, co i tak jest mało prawdopodobne by ktoś pracował z takiej odległości.

median = df.loc[df['DistanceFromHome']<20000, 'DistanceFromHome'].median()
df.loc[df.DistanceFromHome > 20000, 'DistanceFromHome'] = np.nan
df.fillna(median, inplace=True)

#W kolumnie 'Attrition' warto zmienić dane typu 'object' na 'int'.

df['Attrition'] = df['Attrition'].map(
                   {'Yes':1 ,'No':0})

print(df)

#Sprawdzenie na heatmapie korelacji kolumn'.

plt.rcParams["figure.figsize"] = [24, 20]
sns.heatmap(df.corr(), annot=True)
# plt.show()

#Usunięcie jednych z par skorelowanych kolumn.

df = df.drop(['EmployeeCount', 'StandardHours', 'MonthlyIncome', 'JobLevel', 'PercentSalaryHike', 'Over18'], axis=1,)

#Zmienienie danych w taki sposób by były łatwiejsze do przetwarzania przez komputer.

categorical_columns_ls = df.select_dtypes(
    ["object", "category"]
).columns.to_list()

df = pd.get_dummies(
    data=df,
    prefix_sep="_",
    columns=categorical_columns_ls,
    drop_first=True,
    dtype="int8",
)

#Ponowne sprawdzenie wartości odstających za pomocą funkcji.

def outlier_detection(features_series):

    Q1 = features_series.quantile(0.25)
    Q3 = features_series.quantile(0.75)
    IQR = Q3 - Q1
    upper_end = Q3 + 1.5 * IQR
    lower_end = Q1 - 1.5 * IQR
    outlier = features_series[
        (features_series > upper_end) | (features_series < lower_end)
    ]
    return outlier

for column in df.select_dtypes("float64").columns:
    print(f"Outliers w kolumnie {column}")
    outlier = outlier_detection(df[column])
    print(outlier)

#Zmiena danych przy założeniach, ze pracownik w firmie pracowal maksymalnie od 18 do 70 roku życia czyli 52 lata.

median = df.loc[df['YearsAtCompany']<52, 'YearsAtCompany'].median()
df.loc[df.YearsAtCompany > 52, 'YearsAtCompany'] = np.nan
df.fillna(median, inplace=True)

median = df.loc[df['YearsInCurrentRole']<52, 'YearsInCurrentRole'].median()
df.loc[df.YearsInCurrentRole > 52, 'YearsInCurrentRole'] = np.nan
df.fillna(median, inplace=True)

median = df.loc[df['YearsSinceLastPromotion']<52, 'YearsSinceLastPromotion'].median()
df.loc[df.YearsSinceLastPromotion > 52, 'YearsSinceLastPromotion'] = np.nan
df.fillna(median, inplace=True)

median = df.loc[df['YearsWithCurrManager']<52, 'YearsWithCurrManager'].median()
df.loc[df.YearsWithCurrManager > 52, 'YearsWithCurrManager'] = np.nan
df.fillna(median, inplace=True)

median = df.loc[df['TotalWorkingYears']<70, 'TotalWorkingYears'].median()
df.loc[df.TotalWorkingYears > 70, 'TotalWorkingYears'] = np.nan
df.fillna(median, inplace=True)

#Ponowne sprawdzenie wartości odstających.

for column in df.select_dtypes("float64").columns:
    print(f"Outliers w kolumnie {column}")
    outlier = outlier_detection(df[column])
    print(outlier)

#Ponowne sprawdzenie koleracji na heatmapie.

plt.rcParams["figure.figsize"] = [24, 20]
sns.heatmap(df.corr(), annot=True)
#plt.show()

#Powne usunięcie zbyt skolerowanych kolumn.

df = df.drop(['MaritalStatus_Married', 'Department_Sales','JobRole_Sales Executive', 'YearsInCurrentRole','YearsWithCurrManager', 'BusinessTravel_Travel_Rarely', 'StockOptionLevel',  ], axis=1,)

print('table:\n', df)


#Podział danych na treningowe i testowe.

X = df.drop("Attrition", axis=1)
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train dimension= ", X_train.shape)
print("X_test dimension= ", X_test.shape)
print("y_train dimension= ", y_train.shape)
print("y_train dimension= ", y_test.shape)
X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)

#Stworzenie modelu klasyfikacji i sprawdzenie rezultatu predykcji.

knn = KNeighborsClassifier(n_neighbors=5, metric='chebyshev')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

def calculate_metrics(target, prediction, average='weighted'):
    accuracy = accuracy_score(target, prediction)
    precision = precision_score(target, prediction, average=average)
    recall = recall_score(target, prediction, average=average)
    f1 = f1_score(target, prediction, average=average)
    mislabeled = (target != prediction).sum()
    total = len(target)
    return accuracy, precision, recall, f1, mislabeled, total

def print_results(metrics, classifier_id='classifier'):
    print(f'Results for: {classifier_id}')
    print(f'  Accuracy:  {metrics[0]}')
    print(f'  Precision: {metrics[1]}')
    print(f'  Recall:    {metrics[2]}')
    print(f'  F1 score:  {metrics[3]}')
    print(f'  Mislabeled {metrics[4]} out of {metrics[5]}')
    print('\n')

print(print_results(calculate_metrics(y_test, pred), 'kNN'))

