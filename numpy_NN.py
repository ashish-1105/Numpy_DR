#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Reading the CSV files

# In[2]:


data = pd.read_csv("C:/Users/ashis/Downloads/archive/train.csv")
data1 = pd.read_csv("C://Users/ashis/Downloads/test/test.csv")


# # Pre-processing data

# In[3]:


data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255


data1 = np.array(data1)
m,n = data.shape
np.random.shuffle(data1)

X_test = data1.T/255


# # Helper functions

# In[4]:


def init_params():
    w1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    w2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return w1,b1,w2,b2

def relu(n):
    return np.maximum(0,n)

def softmax(n):
    a = np.exp(n) / sum(np.exp(n))
    return a

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size,Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def derive_relu(n):
    return n>0

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1 = w1 - alpha*dw1
    w2 = w2 - alpha*dw2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2
    return w1,b1,w2,b2


# # Propagation functions

# In[5]:


def forward_prop(w1,b1,w2,b2,X):
    z1 = w1.dot(X)+b1
    a1 = relu(z1)
    z2 = w2.dot(a1)+b2
    a2 = softmax(z2)
    return z1,a1,z2,a2

def backward_prop(z1,a1,z2,a2,w2,X,Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dz2 = a2 - one_hot_Y
    dw2 = (dz2.dot(a1.T))/m
    db2 = np.sum(dz2)/m
    dz1 = (w2.T).dot(dz2) * derive_relu(z1)
    dw1 = (dz1.dot(X.T))/m
    db1 = np.sum(dz1)/m
    return dw1,db1,dw2,db2


# # Gradient descent

# In[6]:


def get_predictions(n):
    return np.argmax(n,0)

def get_accuracy(n,m):
    print(n,m)
    return sum(n==m)/m.size

def grad_desc(X,Y,iterations,alpha):
    w1,b1,w2,b2 = init_params()
    for i in range(iterations):
        z1,a1,z2,a2 = forward_prop(w1,b1,w2,b2,X)
        dw1,db1,dw2,db2 = backward_prop(z1,a1,z2,a2,w2,X,Y)
        w1,b1,w2,b2 = update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
        if(i%10==0):
            print("iterations : ", i)
            print("accuracy : ", get_accuracy(get_predictions(a2),Y))
    return w1,b1,w2,b2


# # Model training

# In[7]:


w1,b1,w2,b2 = grad_desc(X_train,Y_train,2000,0.5)


# # Functions for testing predictions

# In[8]:


def make_predictions(X, w1, b1, w2, b2):
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, X)
    predictions = get_predictions(a2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_test[:, index, None], w1, b1, w2, b2)
    print("Prediction: ", prediction)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# # Driver code

# In[9]:


while True:
    x = (input("Choose a number to be tested [press q to quit] : "))
    if(x=='q'):
        break
    x = int(x)
    test_prediction(x,w1,b1,w2,b2)

