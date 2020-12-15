import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


##Uploading Data
path = "D:/travail/Ensae/python datascientist/DataBase/HousingDatabase.csv"

df = pd.read_csv(path, header=None)
df.head()
df.drop(0, axis=1, inplace=True)


##Preprocessing - Exploring dataset

#renaming column
column_name = list(df.ix[0])
df.drop([0], inplace = True)
df.columns = column_name

df.drop(["Id"], axis=1, inplace = True)

#Deleting column with low information
Low_info = []
for column in list(df.columns):
    isna = df[column].isna().sum()/df.shape[0]
    print("{} : ".format(column), isna)

    if isna > 0.15:
        df.drop([column], axis=1, inplace = True)
        Low_info.append(column)

print("Dropping : ", Low_info)
#Dropping :  ['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']


#Correct data format
ListIntFormat = ["LotArea", "YearBuilt", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "PoolArea", "YrSold", "SalePrice"]
df[ListIntFormat] = df[ListIntFormat].astype("int")
print(df.dtypes)


#Exploring data content
for column in list(df.columns):
    print(df[column].value_counts().to_frame())


#Dropping Categories columns, faintly correlated (using boxplot)

sns.boxplot(x="Street", y="SalePrice", data=df)
plt.show()
df.drop(["Street"], axis=1, inplace=True)


sns.boxplot(x="Utilities", y="SalePrice", data=df)
plt.show()
df.drop(["Utilities"], axis=1, inplace=True)


sns.boxplot(x="LandSlope", y="SalePrice", data=df)
plt.show()
df.drop(["LandSlope"], axis=1, inplace=True)


sns.boxplot(x="LotShape", y="SalePrice", data=df)
plt.show()
#Anova test:
grouped_test=df[['LotShape', 'SalePrice']].groupby(['LotShape'])
f_val, p_val = stats.f_oneway(grouped_test.get_group("Reg")['SalePrice'], grouped_test.get_group("IR1")['SalePrice'])
print( "ANOVA results: F=", f_val, ", P =", p_val)
#We only test Reg and IR1, since most of the lots belong to these category. F est quite high, and p-value very low, so we keep this feature. However, we can regroup IR1, IR2, IR3 :
df["LotShape"] = df["LotShape"].map(lambda x: dict(Reg="Reg", IR1="IR", IR2="IR", IR3="IR")[x], df.LotShape.values.tolist())


sns.boxplot(x="BldgType", y="SalePrice", data=df)
plt.show()
df.drop(["BldgType"], axis=1, inplace=True)


sns.boxplot(x="RoofStyle", y="SalePrice", data=df)
plt.show()
df.drop(["RoofStyle"], axis=1, inplace=True)


sns.boxplot(x="HeatingQC", y="SalePrice", data=df)
plt.show()
#Anova test:
grouped_test=df[['HeatingQC', 'SalePrice']].groupby(['HeatingQC'])
f_val, p_val = stats.f_oneway(grouped_test.get_group("Ex")['SalePrice'], grouped_test.get_group("TA")['SalePrice'], grouped_test.get_group("Gd")['SalePrice'], grouped_test.get_group("Fa")['SalePrice'], grouped_test.get_group("Po")['SalePrice'])
print( "ANOVA results: F=", f_val, ", P =", p_val)
#Anova test without "Ex":
grouped_test=df[['HeatingQC', 'SalePrice']].groupby(['HeatingQC'])
f_val, p_val = stats.f_oneway( grouped_test.get_group("TA")['SalePrice'], grouped_test.get_group("Gd")['SalePrice'], grouped_test.get_group("Fa")['SalePrice'], grouped_test.get_group("Po")['SalePrice'])
print( "ANOVA results: F=", f_val, ", P =", p_val)
#We can simplify this feature into two categories, namely "Ex" and the others:
df["HeatingQC"] = df["HeatingQC"].map(lambda x: dict(Ex="Ex", TA="other", Fa="other", Gd="other", Po="other")[x], df.HeatingQC.values.tolist())


sns.boxplot(x="BsmtFullBath", y="SalePrice", data=df)
plt.show()
df.drop(["BsmtFullBath"], axis=1, inplace=True)


sns.boxplot(x="BsmtHalfBath", y="SalePrice", data=df)
plt.show()
df.drop(["BsmtHalfBath"], axis=1, inplace=True)


sns.boxplot(x="HalfBath", y="SalePrice", data=df)
plt.show()
df.drop(["HalfBath"], axis=1, inplace=True)


sns.boxplot(x="BedroomAbvGr", y="SalePrice", data=df)
plt.show()
grouped_test=df[['BedroomAbvGr', 'SalePrice']].groupby(['BedroomAbvGr'])
f_val, p_val = stats.f_oneway( grouped_test.get_group(2)['SalePrice'], grouped_test.get_group(3)['SalePrice'], grouped_test.get_group(4)['SalePrice'])
print( "ANOVA results: F=", f_val, ", P =", p_val)
#We don't delete this feature (F quite high)


sns.boxplot(x="KitchenAbvGr", y="SalePrice", data=df)
plt.show()
df.drop(["KitchenAbvGr"], axis=1, inplace=True)


sns.boxplot(x="Fireplaces", y="SalePrice", data=df)
plt.show()
df["Fireplaces"] = df["Fireplaces"].map(lambda x: dict([(0,0), (1,1), (2,1), (3,1)])[x], df.Fireplaces.values.tolist())


sns.boxplot(x="GarageType", y="SalePrice", data=df)
plt.show()
df.drop(["GarageType"], axis=1, inplace=True)


sns.boxplot(x="GarageQual", y="SalePrice", data=df)
plt.show()
df.drop(["GarageQual"], axis=1, inplace=True)


sns.boxplot(x="PoolArea", y="SalePrice", data=df)
plt.show()
df["PoolArea"] = df["PoolArea"].map(lambda x: dict([(0,0), (738,1), (648,1), (576,1), (555,1), (519,1), (512,1), (480,1)])[x], df.PoolArea.values.tolist())


sns.boxplot(x="YrSold", y="SalePrice", data=df)
plt.show()
grouped_test=df[['YrSold', 'SalePrice']].groupby(['YrSold'])
f_val, p_val = stats.f_oneway(grouped_test.get_group(2006)['SalePrice'], grouped_test.get_group(2007)['SalePrice'], grouped_test.get_group(2008)['SalePrice'], grouped_test.get_group(2009)['SalePrice'], grouped_test.get_group(2010)['SalePrice'])
print( "ANOVA results: F=", f_val, ", P =", p_val)
df.drop(["YrSold"], axis=1, inplace=True)



#Exploring quantitativ features

plt.close()
sns.regplot(x="LotArea", y="SalePrice", data=df)
plt.show()
print(df[["LotArea", "SalePrice"]].corr())

plt.close()
sns.regplot(x="YearBuilt", y="SalePrice", data=df)
plt.show()
print(df[["YearBuilt", "SalePrice"]].corr())

#Maybe a polynomial regression (degree at least higher than 2)

plt.close()
sns.regplot(x="TotalBsmtSF", y="SalePrice", data=df)
plt.show()
print(df[["TotalBsmtSF", "SalePrice"]].corr())


plt.close()
sns.regplot(x="1stFlrSF", y="SalePrice", data=df)
plt.show()
print(df[["1stFlrSF", "SalePrice"]].corr())

#The distribution is not homoscedastic

plt.close()
sns.regplot(x="2ndFlrSF", y="SalePrice", data=df)
plt.show()
print(df[["2ndFlrSF", "SalePrice"]].corr())
#Maybe we should split the dbb in two categories : with and without a second floor

plt.close()
sns.regplot(x="GrLivArea", y="SalePrice", data=df)
plt.show()
print(df[["GrLivArea", "SalePrice"]].corr())
#The distribution is not homoscedastic

plt.close()
sns.regplot(x="GarageArea", y="SalePrice", data=df)
plt.show()
print(df[["GarageArea", "SalePrice"]].corr())

plt.close()
sns.regplot(x="WoodDeckSF", y="SalePrice", data=df)
plt.show()
print(df[["WoodDeckSF", "SalePrice"]].corr())
#faintly correlated


#Significance of features, using p-value
def significance(feature):
    pearson_coef, p_value = stats.pearsonr(df['{}'.format(feature)], df['SalePrice'])
    print("The Pearson Correlation Coefficient of '{}' is".format(feature), pearson_coef, " with a P-value of P =", p_value)

significance("YearBuilt")
significance("LotArea")
significance("TotalBsmtSF")
significance("1stFlrSF")
significance("2ndFlrSF")
significance("GrLivArea")
significance("GarageArea")
significance("WoodDeckSF")
#Everytime, we have a low p-value, so it's highly significant, statistically speaking


#Normalization of quantitativ features
DictIntFormat={}
for column in df.columns:
    if str(df[column].dtypes) == "int32" or str(df[column].dtypes) == "int64":
        DictIntFormat[column]=df[column].max()
        df[column]=df[column]/df[column].max()


#Dummy Variable
def get_dummyvariable(column):
    dummy_variable = pd.get_dummies(df[column])

    list_name = list(dummy_variable.columns)
    dict_name = {}
    for name in list_name:
        dict_name[name] = "{0}_{1}".format(column,name)

    dummy_variable.rename(columns=dict_name, inplace=True)

    return dummy_variable

dummy_MSZoning = get_dummyvariable("MSZoning")
df = pd.concat([df, dummy_MSZoning], axis=1)
df.drop("MSZoning", axis = 1, inplace=True)

dummy_LotShape = get_dummyvariable("LotShape")
df = pd.concat([df, dummy_LotShape], axis=1)
df.drop("LotShape", axis = 1, inplace=True)

dummy_Neighborhood = get_dummyvariable("Neighborhood")
df = pd.concat([df, dummy_Neighborhood], axis=1)
df.drop("Neighborhood", axis = 1, inplace=True)

dummy_Foundation = get_dummyvariable("Foundation")
df = pd.concat([df, dummy_Foundation], axis=1)
df.drop("Foundation", axis = 1, inplace=True)

dummy_BsmtQual = get_dummyvariable("BsmtQual")
df = pd.concat([df, dummy_BsmtQual], axis=1)
df.drop("BsmtQual", axis = 1, inplace=True)

dummy_HeatingQC = get_dummyvariable("HeatingQC")
df = pd.concat([df, dummy_HeatingQC], axis=1)
df.drop("HeatingQC", axis = 1, inplace=True)

dummy_KitchenQual = get_dummyvariable("KitchenQual")
df = pd.concat([df, dummy_KitchenQual], axis=1)
df.drop("KitchenQual", axis = 1, inplace=True)


df["CentralAir"] = df["CentralAir"].map(lambda x: dict(Y=1, N=0)[x], df.CentralAir.values.tolist())

df["Fireplaces"] = df["Fireplaces"].map(lambda x: dict([(1.0,1), (0.0,0)])[x], df.Fireplaces.values.tolist())


#Uploading dataset preprocessed
#df.to_csv("HousingDatabase_preprocessed.csv")


##Model development - IN-SAMPLE

#SIMPLE LINEAR REGRESSION
lm = LinearRegression()

X = df[['YearBuilt']]
Y = df['SalePrice']

lm.fit(X,Y)

print(lm.intercept_)   #Intercept
print(lm.coef_)        #Slop

print('The R-square is: ', lm.score(X, Y))
#We can say that ~ 28.729% of the variation of the price is explained by this simple linear model

Yhat=lm.predict(X)
mse = mean_squared_error(df['SalePrice'], Yhat)
print('The mean square error of price and predicted value is: ', mse)




#MULTIPLE LINEAR REGRESSION
lm2 = LinearRegression()

Z = df[['YearBuilt', 'FullBath', 'BedroomAbvGr']]
Y = df['SalePrice']

lm2.fit(Z,Y)

print(lm2.intercept_)
print(lm2.coef_)

Y_hat2 = lm2.predict(Z)

print('The R-square is: ', lm2.score(Z, df['SalePrice']))

print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['SalePrice'], Y_hat2))

#Visualization of multiple linear regression
width = 12
height = 10
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['SalePrice'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat2, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('SalePrice')
plt.ylabel('Proportion of housing')

plt.show()
plt.close()
#The multiple linear regression with 3 features is not that efficient




#MULTIPLE LINEAR REGRESSION : with all variables
lm3 = LinearRegression()

List_features = list(df.columns)
List_features.remove("SalePrice")

Z3 = df[List_features]
Y = df['SalePrice']

lm3.fit(Z3,Y)

print(lm3.intercept_)
print(lm3.coef_)

Y_hat3 = lm3.predict(Z3)

print('The R-square is: ', lm3.score(Z3, df['SalePrice']))

print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['SalePrice'], Y_hat3))

'''
The output :
The R-square is:  0.8359410500070197
The mean square error of price and predicted value using multifit is:  0.0018151574284287872

The output is quite well : we still need to test our model with other data
We're going to split our dataset (see in the next paragraph
'''


#POLYNOMIAL REGRESSION, on the feature "YearBuilt"

plt.close()
sns.regplot(x="YearBuilt", y="SalePrice", data=df)
plt.show()
print(df[["YearBuilt", "SalePrice"]].corr())

y_data = df['SalePrice']
x_data=df.drop('SalePrice',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['YearBuilt']])
x_test_pr = pr.fit_transform(x_test[['YearBuilt']])

poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat_poly = poly.predict(x_test_pr)
yhat_poly[0:5]

print("Predicted values:", yhat_poly[0:4])
print("True values:", y_test[0:4].values)


PollyPlot(x_train[['YearBuilt']], x_test[['YearBuilt']], y_train, y_test, poly,pr)
#Don't work as expected...

##Machine learning

List_features = list(df.columns)
List_features.remove("SalePrice")
x_data = df[List_features]

y_data = df["SalePrice"]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#Simple Linear Regression

lm4=LinearRegression()
lm4.fit(x_train[['YearBuilt']], y_train)
print("Score on test data : ", lm4.score(x_test[['YearBuilt']], y_test))
print("Score on train data : ", lm4.score(x_train[['YearBuilt']], y_train))
"""
Score on test data :  0.2857069033800925
Score on train data :  0.2716856243784258
"""

#Multiple Linear Regression

lm5=LinearRegression()
lm5.fit(x_train, y_train)
print("Score on test data : ", lm5.score(x_test, y_test))
print("Score on train data : ", lm5.score(x_train, y_train))
"""
Score on test data :  0.8763384752225573
Score on train data :  0.8303516081927521
"""

Y_hat5_train = lm5.predict(x_train)
Y_hat5_test = lm5.predict(x_test)


#Visualization of multiple linear regression
width = 12
height = 10
plt.figure(figsize=(width, height))

ax1 = sns.distplot(y_train, hist=False, color="r", label="Actual TRAIN Value")
sns.distplot(Y_hat5_train, hist=False, color="b", label="Fitted TRAIN Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price, for TRAIN dataset')
plt.xlabel('SalePrice')
plt.ylabel('Proportion of housing')
plt.show()
plt.close()

ax2 = sns.distplot(y_test, hist=False, color="r", label="Actual TEST Value")
sns.distplot(Y_hat5_test, hist=False, color="b", label="Fitted TEST Values" , ax=ax2)
plt.title('Actual vs Fitted Values for Price, for TEST dataset')
plt.xlabel('SalePrice')
plt.ylabel('Proportion of housing')
plt.show()
plt.close()

#The visualisation give us an insight of how much we could be confident on our prediction. It's a quite good model.


#However, we haven't got enough test data (barely 146) : so we should use CrossValidation



#CROSS-VALIDATION SCORE

Rcross = cross_val_score(lm5, x_data, y_data, cv=4)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

yhat5 = cross_val_predict(lm5,x_data, y_data,cv=4)









##Fonction annexe

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('SalePrice')
    plt.legend()
    plt.show()





















