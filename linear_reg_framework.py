'''
Daniel Flores Rodríguez
A01734184
Last modify: September 07 2022

Machine learning Linear regresion using a framework
'''


#we import libryries
import pandas as pd #we use pandas to manage dataframes
import matplotlib.pyplot as plt # and matplotlib to show predictions
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score




'''
we are using a dataset that  contains two columns. 
that is the number of hours student studied and the marks they got. 

'''

data = pd.read_csv('./score.csv')

#we asigned our x and y values, being x our independent variable (hours of study) and y our dependent variable (marks) 
#we use reshape function in order to redimention our y and x variable arrays

X = np.array(data.iloc[:,0]).reshape(-1,1)
Y = np.array(data.iloc[:,1]).reshape(-1,1)

# we can take a look to our data distribution with scatter plot
# plt.scatter(X,Y)
# plt.show()

#building the model

# in order to calculate linear regresion, we need to split the data into trining and testing data
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.25, random_state = 45)

regr = LinearRegression()
regr.fit(x_train,y_train)

#exploring our results

print("Score: ")
print(regr.score(x_test,y_test))

print("coefficient: ")
print(regr.coef_)

print("Intercept: ")
print(regr.intercept_)

#we make some predictions

print("")
print("")


print("Predictions: ")
predict_regr = [[10],[8.5],[6],[4]]

for i in predict_regr:
    print(f"Horas estudiadas {i} puntaje estimado {regr.predict([i])}")

#we calculate the prediction error in test and train

y_pred = regr.predict(x_test)
pred_err_test = y_test - y_pred
y_pred_train = regr.predict(x_train)
pred_err_train = y_train - y_pred_train


print("")
print("")


print("MSE: ", mean_squared_error(y_test, y_pred))
print("Score: ", regr.score(x_test, y_test))


#IMPROVING MODEL WITH RIDGE

clf = Ridge(alpha=1.0)
clf.fit(x_train, y_train)

train_ridge_pred = clf.predict(x_train)

print("")
print("")


print("MSE in ridge train: ", mean_squared_error(y_train, train_ridge_pred))
print("ridge score train: ", r2_score(y_train, train_ridge_pred))
 
test_ridge_pred = clf.predict(x_test)
print("MSE in ridge test: ", mean_squared_error(y_test, test_ridge_pred))
print("ridge score test: ", r2_score(y_test, test_ridge_pred))



#plotting...

figure, axis = plt.subplots(2,3)

#using test

#Lineal regression
axis[0,0].scatter(x_test, y_test, alpha = 0.9)
axis[0,0].plot(x_test, y_pred, color='red')
axis[0,0].set_title("study hours vs score (test data)")
axis[0,0].set(xlabel = 'study hours', ylabel = 'score')

#Histograms
axis[0,1].hist(pred_err_test, edgecolor = 'black')
axis[0,1].set_title('Histogram of test prediction error')
axis[0,1].set(xlabel = 'Prediction error (Y_test - Y_pred)', ylabel = 'Frequency')

#Var
axis[0,2].scatter(x_test, y_test, alpha = 0.9, label = 'Real data')
axis[0,2].scatter(x_test, pred_err_test, color='red',alpha = 0.9, label = 'Predicted data')
axis[0,2].set_title("Real test data vs Predicted test data")
axis[0,2].set(xlabel = 'study hours', ylabel = 'score')
axis[0,2].legend()

#using train

#regresion
axis[1,0].scatter(x_train, y_train, alpha = 0.9)
axis[1,0].plot(x_train, y_pred_train, color='red')
axis[1,0].set_title("study hours vs score")
axis[1,0].set(xlabel = 'study hours', ylabel = 'score')

#histogram
axis[1,1].hist(pred_err_train, edgecolor = 'black')
axis[1,1].set_title('Histogram of train prediction error')
axis[1,1].set(xlabel = 'Prediction error (Y_train - Y_pred_train)', ylabel = 'Frequency')

#var
axis[1,2].scatter(x_train, y_train, alpha = 0.9, label = 'Real data')
axis[1,2].scatter(x_train, pred_err_train, color='red',alpha = 0.9, label = 'Predicted data')
axis[1,2].set_title("Real train data vs Predicted train data")
axis[1,2].set(xlabel = 'study hours', ylabel = 'score')
axis[1,2].legend()
plt.show()






