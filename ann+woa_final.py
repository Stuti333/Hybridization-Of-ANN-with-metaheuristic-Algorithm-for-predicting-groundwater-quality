# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:45:35 2023

@author: Nupur
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Ground Water .csv')
#print(data)

# Handle missing values
data = data.fillna(data.mean())
#print(data)

# Handle outliers
data = data.clip(lower=data.quantile(0.01), upper=data.quantile(0.99), axis=1)

# Encode categorical variables
data = pd.get_dummies(data)

# Scale numerical variables
scaler = StandardScaler()
data[data.select_dtypes(include=['float64']).columns] = scaler.fit_transform(data.select_dtypes(include=['float64']))


data = data.drop(data.columns[[1, 2]], axis=1)
X = data.iloc[:,:-3]
y = data.iloc[:, -1]
#print("\nX")
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#print("\nx: ")
print("\nx:",X_train)
#print(X_train.shape())
print("\ny: ")
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Choose a model to use
model = LinearRegression()

# Split the dataset into k-folds
kf = KFold(n_splits=5, shuffle=True)

# Perform cross-validation
mse_scores = []
for train_index, test_index in kf.split(X_train):
    # Split the data into training and testing sets for this fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the model on the training set and test it on the testing set
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)
    mse = mean_squared_error(y_test_fold, y_pred)
    mse_scores.append(mse)
   
# Compute the average mean squared error across all folds
avg_mse = sum(mse_scores) / len(mse_scores)
#print("\navgmse: ",avg_mse)



Y_train=np.unique(y_train_fold)
#y_train=y_train.reshape(-1,1)
#print("\nunique: ",y_train)
#print("\nunique shape: ",y_train.shape)
#print("\nlength unique shape: ",len(y_train))


#from woa import WOA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from functools import partial
import numpy as np


def f(X):
    A = 10
    sol = []
    for ind in X:
        sol.append(A*len(ind) + sum([(i**2 - A * np.cos(2 * np.pi * i)) for i in ind]) )#output-Y

    return np.array(sol)
x_lb=y_lb=-500
x_ub=y_ub=500








class WOA:
    def __init__(self, obj_func, n_whale, spiral_constant, n_iter,lb, ub,W):
        self.obj_func = obj_func
        self.n_whale = n_whale
        self.spiral_constant = spiral_constant
        self.n_iter = n_iter
        #print('--------------------')
        self.whale = {}
        self.prey = {}
        self.W=W
        #print('----------------------------')
        self.lb = np.array([x_lb, y_lb])

        self.ub = np.array([x_ub, y_ub])

    def init_whale(self):
        tmp = [np.random.uniform(self.lb, self.ub, size=(len(self.lb),))
             for i in range(self.n_whale)]
        print("\n temp:",tmp)
        self.whale['position'] = np.array(tmp)
        self.whale['fitness'] = self.obj_func(self.whale['position'])

    def init_prey(self):
        
        tmp = [np.random.uniform(self.lb, self.ub, size=(len(self.lb),))]
       
        self.prey['position'] = np.array(tmp)
        self.prey['fitness'] = self.obj_func(self.prey['position'])

   
    def update_prey(self):
        if self.whale['fitness'].min() < self.prey['fitness'][0]:
            self.prey['position'][0] = self.whale['position'][self.whale['fitness'].argmin()]
            self.prey['fitness'][0] = self.whale['fitness'].min()

    def search(self, idx, A, C):
        random_whale = self.whale['position'][np.random.randint(low=0, high=self.n_whale,
                                                                size=len(idx[0]))]
        d = np.abs(C[..., np.newaxis] * random_whale - self.whale['position'][idx])
        self.whale['position'][idx] = np.clip(random_whale - A[..., np.newaxis] * d, self.lb, self.ub)

    def encircle(self, idx, A, C):
        #d = np.abs(C[..., np.newaxis] * self.prey['position'].reshape(1, -1) - self.whale['position'][idx])
        d = np.abs(np.reshape(C, (-1, 1)) * self.prey['position'].reshape(1, -1) - self.whale['position'][idx])

        self.whale['position'][idx] = np.clip(self.prey['position'][0] - A[..., np.newaxis] * d, self.lb, self.ub)

    def bubble_net(self, idx):
        d_prime = np.abs(self.prey['position'] - self.whale['position'][idx])
        l = np.random.uniform(-1, 1, size=len(idx[0]))
        self.whale["position"][idx] = np.clip(
            d_prime * np.exp(self.spiral_constant * l)[..., np.newaxis] * np.cos(2 * np.pi * l)[..., np.newaxis]
            + self.prey["position"],
            self.lb,
            self.ub,
        )

    def optimize(self, a):

        p = np.random.random(self.n_whale)
        r1 = np.random.random(self.n_whale)
        r2 = np.random.random(self.n_whale)
        A = 2 * a * r1 - a
        C = 2 * r2
        search_idx = np.where((p < 0.5) & (abs(A) > 1))
        encircle_idx = np.where((p < 0.5) & (abs(A) <= 1))
        bubbleNet_idx = np.where(p >= 0.5)
        self.search(search_idx, A[search_idx], C[search_idx])
        self.encircle(encircle_idx, A[encircle_idx], C[encircle_idx])
        self.bubble_net(bubbleNet_idx)
        self.whale['fitness'] = self.obj_func(self.whale['position'])

    def run(self):
        self.init_whale()
        self.init_prey()
        f_values = [self.prey['fitness'][0]]
        #print("\n\n\n\n\noptimal sol: ",self.prey['position'][0])
        for n in range(self.n_iter):
            #print("Iteration = ", n, " f(x) = ", self.prey['fitness'][0])
            a = 2 - n * (2 / self.n_iter)
            self.optimize(a)
            self.update_prey()
            #l.append((self.loss(out, y_wt)))
            #acc.append(abs((1-(sum(l)/len(x)))*10))
            f_values.append(self.prey['fitness'][0])
           
        optimal_x = self.prey['position'].squeeze()
        #print("\n f_val: ",f_values)
        #print("\n optimal: ",optimal_x)
        return f_values, optimal_x






#neural Network
input_layer_size = X_train.shape[1]
print("\nils: ",X_train.shape)

import numpy as np

class NeuralNetwork:
    def __init__(self,input_layer_size,hidden_layer_size,output_layer_size,X):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
       
        # Initialize the weights with random values
        self.W1 = np.random.randn(input_layer_size,hidden_layer_size)
        self.W2 = np.random.randn(hidden_layer_size,output_layer_size)
        #print("\nW1111: ",self.W1)
        #print("\nw1_size: ",self.W1.shape)
        #print("\nW2222: ",self.W2)
        #print("\nw2_size: ",self.W2.shape)
        # Initialize the biases with zeros
        self.b1 = np.random.randn(len(X),hidden_layer_size)#x_train
        self.b2 = np.random.randn(len(X),output_layer_size)#x_train
        #print("\nB1:",self.b1)
        #print("\nB2:",self.b2)
        #print("\nB2size: ",len(self.b2))
   
    def sigmoid(self,x):
        return(1/(1 + np.exp(-x)))
       
    def forward_propagation(self, X):
        # Calculate the hidden layer activations
       
        #self.b1=np.tile(self.b1,(94,1))
       
        #X_T=np.transpose(X)
        self.Z1 =np.dot(X,self.W1) + self.b1
        #print("\nZ1: ",self.Z1)
        #print("\nz1size: ",self.Z1.shape)
        self.A1 = self.sigmoid(self.Z1)
        #print("\nA1: ",self.A1)
        #print("\nA1size: ",self.A1.shape)
        # Calculate the output layer activations
        self.Z2 = np.dot(self.A1,self.W2) + self.b2
        #print("\nZ2: ",self.Z2)
        #print("\nZ2size: ",self.Z2.shape)
        self.A2 = self.sigmoid(self.Z2)
        #print("/na2l: ",self.A2.shape)
        #print("\nA2: ",self.A2)
        #print("\n A2 ka shape",self.A2.shape)
        return self.A2
   
    def generate_wt(self,x, y):
        l =[]
        for i in range(len(x) * len(y)):
            l.append(np.random.randn())
        return(np.array(l).reshape(len(x),len(y)))

    def convert(self):
       #print("\nwwww1 shp: ",self.W1.shape)
       #print("\nwwww2 shp: ",self.W2.shape)
       
       #print("\n print wwwwwwwwwwwwwwwwwwwwwsize: ",self.W1.shape)
       min_val=self.W1[0][0]
       row=len(self.W1)
       col=len(self.W1[0])
       #arr2=self.W1[row-1]
       max_val=self.W1[-1][-1]
       #print("---------------------------------------------------")
       print("\nmin,max w1",min_val,max_val)
       
       #print("\n print wwwwwwwwwwwwwwwwwwwwwsize: ",self.W2.shape)
       min_val1=self.W2[0][0]
       row=len(self.W2)
       col=len(self.W2[0])
       #arr2=self.W2[row-1]
       max_val1=self.W2[-1][-1]
       #print("---------------------------------------------------")
       #print("\nmin,max w2",min_val1,max_val1)
       woa = WOA(f, 2, 0.5, 10, min_val, max_val,self.W1)
       f_values1, self.W1 = woa.run()
       print("\n\nw1: ",self.W1)
       #plt.subplot(1, 3, 1)
       #plt.plot(f_values1)
       #plt.title("Fitness Values over Time")
       #plt.xlabel("Iteration")
       #plt.ylabel("Fitness")
       
       woa = WOA(f, 2, 0.5, 10, min_val1, max_val1,self.W2)
       f_values2, self.W2 = woa.run()
       print("\n\nw2: ",self.W2)
       #plt.subplot(1, 3, 3)
       #plt.plot(f_values2)
       #plt.title("Fitness Values over Time")
       #plt.xlabel("Iteration")
       #plt.ylabel("Fitness")
       return f_values1,f_values2



    def backward_propagation(self, X, Y, output,learning_rate):
        # Calculate the error in the output layer
        #print("\nY:",Y)
        #print("\ny_shape: ",Y.shape)
        #print("\noutput: ",output.shape)
        row1,col1=self.W1.shape
        row2,col2=self.W2.shape
        self.W1=np.array(self.W1)
        np.sort(self.W1, axis=1)
        self.W2=np.array(self.W2)
        np.sort(self.W2, axis=1)
        #self.W1=self.W1.flatten(order='C')
        #self.W2=self.W2.flatten(order='C')
        size1=self.W1.size
        #print("\nsize1: ",size1)
        size2=self.W2.size
        #print("\nsize2: ",size2)
        f1_values,f2_values=self.convert()
        minw1=self.W1[1]
        maxw1=self.W1[0]
        #print("\nminw1: ",minw1)
        #print("\nmaxw1: ",maxw1)
        self.W1=np.random.uniform(minw1, maxw1, size1)
        self.W1=np.array(self.W1)
        self.W1 = np.reshape(self.W1, (row1, col1))
        minw2=self.W2[1]
        maxw2=self.W2[0]
        #print("\nminw2: ",minw2)
        #print("\nmaxw2: ",maxw2)
        self.W2=np.random.uniform(minw2, maxw2, size2)
        #print("\nabc:",self.W1)
        #print("\ndef: ",self.W2)
        self.W2=np.array(self.W2)
        self.W2 = np.reshape(self.W2, (row2, col2))
        #print("\n self.W2",self.W2.shape)
        #print("\n self.W1",self.W1.shape)
        #print("\n self.output",output.shape)
        Y=np.array(Y)
        Y=Y.reshape(-1,1)
        #print("\n shh",Y.shape)
       
        dZ2 = output - Y
        #print("\ndz2 ",dZ2)
        #print("\n w2: ",self.W2)    
        dW2 = np.dot(self.A1.T,dZ2)
       
        #print("\ndw2: ",dW2)
        db2 = np.sum(dZ2, axis=1, keepdims=True)
       
       
       # Calculate the error in the hidden layer
        dZ1=np.multiply((self.W2.dot((dZ2.T))).T,(np.multiply(self.A1, 1-self.A1)))
        #dZ1 = np.dot(self.W2.T, dZ2) * (self.A1 * (1 - self.A1))
        dW1 = np.dot(X.T,dZ1)
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        #print("\ndw1: ",dW1)
        #print("\ndb1: ",db1)
        #print("\ndw2: ",dW2)
        #print("\ndb2: ",db2)
        # Update the weights and biases
        self.W1 -= learning_rate*dW1
        self.b1 -= learning_rate*db1
        self.W2 -= learning_rate*dW2
        self.b2 -= learning_rate*db2
        return f1_values,f2_values
   


    #1
   
    def loss(self,y_pred, y_true):
        y_true = y_true.values.reshape(-1, 1)
        #print("\n y_p:",y_pred)
        #print("\n y_t:",y_true)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        y_true_binary = (y_true >= 0.5).astype(int)
        mse = np.mean((y_pred - y_true_binary)**2)
        print(f"\nMSE: {mse}")
        return mse
        

    
    def accuracy(self,y_pred,y_true):
       y_true = y_true.values.reshape(-1, 1)
       y_pred_binary = (y_pred >= 0.5).astype(int)
       # Convert true labels to binary based on threshold of 0.5
       y_true_binary = (y_true >= 0.5).astype(int)
       # Calculate accuracy as percentage of correct predictions
       return (y_pred_binary == y_true_binary).mean() * 100
    def rmsee(self,y_pred,y_train):
        mse = mean_squared_error(y_train, y_pred)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        #print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        return rmse
   
    
    def train(self,x, Y,  epoch =10,alpha = 0.01):
        acc =[]
        losss =[]
        rm=[]
        for j in range(epoch):
            out = self.forward_propagation(x)
            f1,f2=self.backward_propagation(x, Y,out,alpha)
            #print("\n out: ",out)
            #print("\n Y:",Y)
            print("epochs:", j + 1, "======== acc:", self.accuracy(out, Y))  
            acc.append(self.accuracy(out,Y))
            losss.append(self.loss(out,Y))
            rm.append(self.rmsee(out,Y))
            
        #print("\n rm:",rm)
        return(acc, losss,rm,f1,f2)
   
    
    def predict(self,x):
       out = self.forward_propagation(x)
       #print("\n out: ",out)
       new_arr=[]
       for i in range(len(out)):
           if(out[i][0]<0.05):
               new_arr.append(0)
           else:
               new_arr.append(1)
       #print("\n new_arr: ",new_arr)

# Define your ANN architecture
input_layer_size = X_train.shape[1]

hidden_layer_size = 10
output_layer_size = 1

weights = np.random.rand(input_layer_size*hidden_layer_size + hidden_layer_size*output_layer_size)

def fitness_function(weights):
    
    nn=NeuralNetwork(input_layer_size,hidden_layer_size,output_layer_size,X_train)
    return nn
   

    

   

ff=fitness_function(weights)
val=ff.forward_propagation( X_train)


acc,losss,rm,f1,f2=ff.train(X_train,y_train,10,0.01)
max_accuracy = acc[0]

for i in range(1, len(acc)):
    if acc[i] > max_accuracy:
        max_accuracy = acc[i]

print("Accuracy:", max_accuracy)
print("Loss:",losss[len(losss)-1])
#print("\ntrain: ",acc,losss)
#print(ff.predict(X_train))


#print("\nacc: ",ff.get_accuracy(X_train,y_wt))



#import matplotlib.pyplot as plt1
 
# plotting accuracy
#plt.subplot(1, 3, 3)
#plt.plot(rm)
#plt.ylabel('RMSE value')
#plt.xlabel("Epochs:")
#plt.show()
 

# plotting Loss
plt.subplot(1, 2, 1)
plt.plot(f1)
plt.title("Fitness Values over Time")
plt.xlabel("Iteration")
plt.ylabel("Fitness")

plt.subplot(1, 2, 2)
plt.plot(f2)
plt.title("Fitness Values over Time")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
#plt.plot(losss)
#plt.ylabel('Fitness')
#plt.xlabel("Iteration:")
#plt.show()
