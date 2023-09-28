#Author: Datta Sheregar
#Date of start: Sep 21 2023
#Date of end: Sep 27 2023
#Purpose:
#   Practise with programming a Linear Regression Model by reading data from 
#   a csv file and using statistics to filter outliers and output indicators
#   of the model's performance, while also adding a feature to alter the
#   model from linear to polynomial of x-degree
#Use: 
#   Use the csv file BostonHousing.csv and try changing the polynomial
#   degree at line 53, then observe how the mse, r2, and mae values change
#Pre-requisites:
#   Run command pip install pandas scikit-learn

#Input libraries
import pandas as pd                                                                             #Used for data handling/processing
from sklearn.model_selection import train_test_split                                            #Used for splitting dataset into a train set and test set
from sklearn.preprocessing import PolynomialFeatures, StandardScaler                            #For adding polynomial features and standardizing the input
from sklearn.linear_model import LinearRegression                                               #For creating a linear regression model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score   #For calculating the mse, r2, and mae
import numpy as np                                                                              #For performing few calculations of arrays


#Importing and formatting the data from the csv file
data = pd.read_csv('BostonHousing.csv')                                                                             #Imports the csv file in the same folder
data.dropna(inplace = True)                                                                                         #Removes any blank entries

input_columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']   #Takes these columns from the csv file as inputs
output_columns = ['medv']                                                                                           #Takes the column medv as the output
INPUT = data[input_columns]                                                                                         #Makes a dataset of the input columns
OUTPUT = data[output_columns]                                                                                       #Makes a dataset of the output columns
print(data)
print(INPUT)
print(OUTPUT)
#IQR Section
lower_percentile = 25                                                           #Determines lower percentile for filtering outliers
upper_percentile = 75                                                           #Determines upper percentile for filtering outliers

for column in INPUT:                                                            #For each column from the inputs...
    Q1 = np.percentile(data[column], lower_percentile)                          #Finds the value of the 25th percentile from each input column
    Q3 = np.percentile(data[column], upper_percentile)                          #Finds the value of the 75th percentile from each input colums
    IQR = Q3 - Q1                                                               #The Interquartile Range is the two inner quartile ranges, between the 25th and the 75th

    lower_bound = Q1 - 1.5*IQR                                                  #Find the value from each column that is considered the lower border value
    upper_bound = Q3 + 1.5*IQR                                                  #Find the value from each column that is considered the upper border value

    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)  #Form an array of boolean value where every TRUE is in the same position as an outlier from an input

    data = data[~outlier_mask]                                                  #Rewrite the input colums by removing every TRUE from the outlier mask

#"""
#Modify data (if necessary)
X_train, X_test, Y_train, Y_test = train_test_split(INPUT, OUTPUT, test_size=0.25, random_state = 1)    #Randomly shuffle all the data and reserve 25% of the data for test purposes

poly_features = PolynomialFeatures(degree = 1)                                                          #Use the degree to give polynomial features (degree of 1 is a linear line)
X_train_poly = poly_features.fit_transform(X_train)                                                     #Transform the input train set with polynomial features
X_test_poly = poly_features.transform(X_test)                                                           #Transform the input test set with polynomial features

model = LinearRegression()                                                                              #Generate a Linear Regression Model
model.fit(X_train_poly, Y_train)                                                                        #Load the model with to find a relation between the input and output

#Output statistics
Y_pred = model.predict(X_test_poly)                                                                     #Using the model use the 25% test inputs to predict outcomes
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))                                                      #Calculate the rmse between the predicted and 25% of the real values
r2 = r2_score(Y_test, Y_pred)                                                                           #Calculate the r2 score of the above two sets
mae = mean_absolute_error(Y_test, Y_pred)                                                               #Calculate the mae between the predicted and 25% of the real values
print(f"Root Mean Squared Error: {rmse}")                                                               #Output the rmse
print(f"Mean Absolute Error: {mae}")                                                                    #Output the r2 score
print(f"R-squared: {r2}")                                                                               #Output the mae
#"""