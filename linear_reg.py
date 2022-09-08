'''
Daniel Flores Rodríguez
A01734184
Last modify: August 26 2022

Machine learning Linear regresion algorithm using gradient descendent
'''

#we import libryries
import pandas as pd #we use pandas to manage dataframes
import matplotlib.pyplot as plt # and matplotlib to show predictions


'''
we are using a dataset that  contains two columns. 
that is the number of hours student studied and the marks they got. 

'''

data = pd.read_csv('./score.csv')

#we asigned our x and y values, being x our independent variable (hours of study) and y our dependent variable (marks) 
X = data.iloc[:,0]
Y = data.iloc[:,1]


# we can take a look to our data distribution with scatter plot
#plt.scatter(X,Y)
#plt.show()

# we define our variables that we're going to use in our model
m = 0 # -> stands for the slopeof the line
c = 0 # -> stands for the intercept
L = .0001 # -> stands for the learning rate, aka. alpha 
epochs = 1000 #-> stands for the number of iterarions

#builiding the model



def GradientDesc(m,c,alpha,iterations,x,y):
	'''
		Function that returns m and c i order to calculate predictions

		params:
		m -> slope
		c -> intercept
		alpha -> lerning rate
		iterations -> no. of iterations 
		x -> X values (hours)
		y -> Y values (notes)

	'''
	iter = 0 # variable to manage the while iteration
	
	n = float(len(x)) # variable to calculate the avrg

	while True: 
		Y_pred = m * X + c # we calculate the hyp
		D_m = (-2/n) * sum(x * ( y - Y_pred )) # we calculate derivate of D_m
		D_c = (-2/n) * sum( y - Y_pred) # we calculate derivate of D_c
		m = m -alpha*D_m # we update the value of m and c
		c = c -alpha*D_c
		iter = iter + 1
		if( iter == iterations ): #when the process finish, we print the values and we exit the loop
			print("m:")
			print(m)
			print("c:")
			print(c)
			break
	
	return(m,c)

def Prediction(m,c,x_input):
	y = m*x_input+c
	print(f"Horas estudiadas {x_input} puntaje estimado {y}")

	return




m, c = GradientDesc(m,c,L,epochs,X,Y) # we call the function 

print(f"m: {m} and {c} ")
y_pred = m*X + c # calculate the predicted y


#predictions
Prediction(m,c,10.0)
Prediction(m,c,8.5)
Prediction(m,c,6.0)
Prediction(m,c,4.0)



#we plot in a scatter plot the results
plt.scatter(X,Y)
plt.plot([min(X),max(X)],[min(y_pred),max(y_pred)], color='red')
plt.xlabel("Horas")
plt.ylabel("Puntaje")
plt.title("Horas vs Puntaje")
plt.show()