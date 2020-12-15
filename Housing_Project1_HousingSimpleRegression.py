"""
Il s'agit d'un premier projet, sur une petite base de données "HousingSimpleRegression.csv"
Après le prétraitement, j'ai réalisé des regressions linéaires In-sample.
Un projet plus poussé a été mené sur une plus grosse base de données, ceci se trouve dans le fichier "Housing_Project.py"
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

##Uploading Dataset
path = "D:/travail/Ensae/python datascientist/DataBase/HousingSimpleRegression.csv"

df = pd.read_csv(path, header=None)
df.head()

##Preprocessing

#renaming column
column_name = list(df.ix[0])
df.drop([0], inplace = True)
df.columns = column_name


df.drop(['prefarea'], axis=1, inplace = True)

#Correct data format
df[["price", "area", "bedrooms", "bathrooms", "stories", "parking"]] = df[["price", "area", "bedrooms", "bathrooms", "stories", "parking"]].astype("int")
print(df.dtypes)

#Data normalization
max_price = df["price"].max()
max_area = df["area"].max()

df["price"] = df["price"]/df["price"].max()
df["area"] = df["area"]/df["area"].max()

#Dummy Variable
dummy_variable_1 = pd.get_dummies(df["furnishingstatus"])
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("furnishingstatus", axis = 1, inplace=True)

df["mainroad"] = df["mainroad"].map(lambda x: dict(yes=1, no=0)[x], df.mainroad.values.tolist())

df["guestroom"] = df["guestroom"].map(lambda x: dict(yes=1, no=0)[x], df.mainroad.values.tolist())

df["basement"] = df["basement"].map(lambda x: dict(yes=1, no=0)[x], df.mainroad.values.tolist())

df["hotwaterheating"] = df["hotwaterheating"].map(lambda x: dict(yes=1, no=0)[x], df.mainroad.values.tolist())

df["airconditioning"] = df["airconditioning"].map(lambda x: dict(yes=1, no=0)[x], df.mainroad.values.tolist())

#save into csv
df.to_csv("HousingSimpleRegression_processed.csv")


##Data exploratory

print(df.shape)
print(df.info())
print(df.describe())

#Missing value
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

df['bedrooms'].value_counts().to_frame()
df['bathrooms'].value_counts().to_frame()
df['stories'].value_counts().to_frame()  #stories=étage
df['mainroad'].value_counts().to_frame()
df['guestroom'].value_counts().to_frame()
df['basement'].value_counts().to_frame() #basement = sous-sol, cave
df['parking'].value_counts().to_frame()
df['prefarea'].value_counts().to_frame()
df['furnishingstatus'].value_counts().to_frame()

df.corr()

#Analysis of continuous numerical variables
print(df[["area", "price"]].corr())
sns.regplot(x="area", y="price", data=df)
plt.show()

#Analysis of categorical variable
sns.boxplot(x="bedrooms", y="price", data=df)
plt.show()

#Means of price, grouping by number of bedrooms
df_group_bedrooms = df[["bedrooms", "price"]].groupby(['bedrooms'],as_index=False).mean()

#Means of price, grouping by number of bathrooms and stories
df_group_bathstor = df[["price", "bathrooms", "stories"]].groupby(['stories','bathrooms'],as_index=False).mean()
df_group_bathstor_pivot = df_group_bathstor.pivot(index="bathrooms", columns="stories")
df_group_bathstor_pivot = df_group_bathstor_pivot.fillna(0)

fig, ax = plt.subplots()
im = ax.pcolor(df_group_bathstor_pivot, cmap='RdBu')
row_labels = df_group_bathstor_pivot.columns.levels[1]
col_labels = df_group_bathstor_pivot.index
ax.set_xticks(np.arange(df_group_bathstor_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_group_bathstor_pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
ax.set_xlabel("stories")
ax.set_ylabel("bathrooms")
plt.xticks(rotation=90)
plt.colorbar(im)
plt.show()

#Significance of features, using p-value
def significance(feature):
    pearson_coef, p_value = stats.pearsonr(df['{}'.format(feature)], df['price'])
    print("The Pearson Correlation Coefficient of '{}' is".format(feature), pearson_coef, " with a P-value of P =", p_value)

#ANOVA test : bedrooms is a categorial features

sns.boxplot(x="bedrooms", y="price", data=df)
plt.show()

grouped_test=df[['bedrooms', 'price']].groupby(['bedrooms'])

#categories grouped
f_val, p_val = stats.f_oneway(grouped_test.get_group(1)['price'], grouped_test.get_group(2)['price'], grouped_test.get_group(3)['price'], grouped_test.get_group(4)['price'], grouped_test.get_group(5)['price'],
grouped_test.get_group(6)['price'])

print( "ANOVA results: F=", f_val, ", P =", p_val)

#separately : bedroom=4 and bedroom=5
f_val, p_val = stats.f_oneway(grouped_test.get_group(4)['price'], grouped_test.get_group(5)['price'])

print( "ANOVA results: F=", f_val, ", P =", p_val)


##Model development - IN-SAMPLE

#SIMPLE LINEAR REGRESSION
lm = LinearRegression()

X = df[['area']]
Y = df['price']

lm.fit(X,Y)

print(lm.intercept_)   #Intercept
print(lm.coef_)        #Slop

print('The R-square is: ', lm.score(X, Y))
#We can say that ~ 28.729% of the variation of the price is explained by this simple linear model

Yhat=lm.predict(X)
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#Visualization of simple linear regression
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="area", y="price", data=df)
plt.ylim(0,)
plt.show()

#Visualization of residual
width = 12
height = 10
plt.figure(figsize=(width, height))

sns.residplot(df['area'], df['price'])
plt.show()
#commentary : We can see from this residual plot that the residuals are  randomly spread around the x-axis, which leads us to believe that a linear model is appropriate for this data.



#MULTIPLE LINEAR REGRESSION
lm2 = LinearRegression()

Z = df[['area', 'bathrooms', 'bedrooms']]
Y = df['price']

lm2.fit(Z,Y)

print(lm2.intercept_)
print(lm2.coef_)

Y_hat2 = lm2.predict(Z)

print('The R-square is: ', lm2.score(Z, df['price']))

print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Y_hat2))

#Visualization of multiple linear regression
width = 12
height = 10
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat2, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of housing')

plt.show()
plt.close()
#We can see that the fitted values are reasonably close to the actual values


#MULTIPLE LINEAR REGRESSION : with all variables
lm3 = LinearRegression()

Z3 = df[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'furnished', 'semi-furnished', 'unfurnished']]
Y = df['price']

lm3.fit(Z3,Y)

print(lm3.intercept_)
print(lm3.coef_)

Y_hat3 = lm3.predict(Z3)

print('The R-square is: ', lm3.score(Z3, df['price']))

print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Y_hat3))
















