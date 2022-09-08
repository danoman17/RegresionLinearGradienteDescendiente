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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



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

print("Predictions: ")
predict_regr = [[10],[8.5],[6],[4]]

for i in predict_regr:
    print(f"Horas estudiadas {i} puntaje estimado {regr.predict([i])}")

#we plotting in a scatter plot the results
'''
plt.scatter(X,Y)
plt.plot([min(X),max(X)],[min(y_test),max(y_test)], color='red')
plt.xlabel("Horas")
plt.ylabel("Puntaje")
plt.title("Horas vs Puntaje")
plt.show()

'''


y_pred = regr.predict(x_test);
y_pred_train = regr.predict(x_train)

#plot de la regresión
figure, axis = plt.subplots(2)

axis[0].scatter(x_test, y_test)
axis[0].plot(x_test, y_pred, color='red')
axis[0].set_title("Study Hours(x) vs Score(y) (test data)")

axis[1].scatter(x_train, y_train)
axis[1].plot(x_train, y_pred_train, color ='red')
axis[1].set_title("Study Hours(x) vs Score(y) (traindata)")

plt.show()






