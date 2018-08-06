from __future__ import print_function,division

import torch


"""
Linear Regression
"""

import numpy as np

from matplotlib import pyplot as plt


device = torch.device('cuda')
dtype = torch.float


"""
#Randomly generated data

np.random.seed(42)
pts = 50
x_vals = np.random.rand(2,50)
xtrain_array = np.asarray(x_vals,dtype=np.float32).reshape(-1,1)
m = 1
alpha = np.random.rand(1)
beta = np.random.rand(1)
ytrain_array = np.asarray([2*i+m for i in x_vals], dtype=np.float32).reshape(-1,1)
"""

def processingCsv(csv_path):
    with open(csv_path) as csv_file:
        csv_list = csv_file.readlines()

        input_list = [value.strip().split(',') for value in csv_list]
        x = []
        y = []
        for value in input_list[1:]:
            x.append(float(value[0]))
            y.append(float(value[1]))

        x_array = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        return x_array,y_array

class LinearRegression(object):
    def __init__(self,learning_rate,w_dim,batch_size):
        self.learning_rate = learning_rate
        self.weight = torch.randn(1,w_dim, device=device, dtype=dtype, requires_grad=True)
        self.bias = torch.randn(1, device=device, dtype=dtype, requires_grad=True)
        self.batch_size = batch_size

    def forward(self,x,y,batch_size):
        y_pred =  self.weight*x + self.bias
        loss = ((y_pred-y).pow(2).sum() ) / batch_size
        return loss

    def backward(self,loss):
        loss.backward()
        with torch.no_grad():
            self.weight-=self.learning_rate * self.weight.grad
            self.bias-= self.learning_rate * self.bias.grad
            self.weight.grad.zero_()
            self.bias.grad.zero_()

    def predict(self,x):
        print(self.weight, self.bias)
        return x*self.weight + self.bias


def run(csv_path,test_csv_path):
    xtrain_array,ytrain_array =  processingCsv(csv_path)
    xtest_array, ytest_array = processingCsv(test_csv_path)
    num_epochs = 2000
    learning_rate = 0.0001
    batch_size =699
    num_feature =1

    linregression = LinearRegression(learning_rate,num_feature,batch_size)
    for epoch  in range(num_epochs):
        for idx in range(0,ytrain_array.shape[0],batch_size):
            x = torch.tensor(xtrain_array[idx:batch_size+idx,:],device=device,dtype=dtype)
            y = torch.tensor(ytrain_array[idx:batch_size+idx,:],device=device,dtype=dtype)
            loss= linregression.forward(x,y,batch_size)
            linregression.backward(loss)
        if (epoch+1) % 100 ==0:
            print('Loss ',epoch+1, loss.item())

    #testing

    y_predict_test = linregression.predict(torch.tensor(xtest_array,device=device,dtype=dtype))
    for idx in range(50):

        print('Actual output is  {:f} ---> Predicted output is  {:f}'.format(y_predict_test[idx,0].item(), ytest_array[idx,0]))


if __name__ =="__main__":
    csv_path = "./data/regression/train_kaggle.csv"
    test_csv_path ="./data/regression/test_kaggle.csv"
    run(csv_path,test_csv_path)