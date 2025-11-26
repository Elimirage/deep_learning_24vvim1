# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 18:55:16 2025

@author: student
"""

import torch 
import numpy as np
import pandas as pd
from random import randint
#Cоздайте тензор x целочисленного типа, хранящий случайное значение.
x = torch.randint(0, 4,(1,),dtype=torch.int32)
print(x) 

#Преобразуйте тензор к типу float32; 
x_float = x.to(dtype=torch.float32)
print(x_float)
x_float.requires_grad=True # Теперь все операци над тензором a будут отслеживаться
print(x_float)

#Проведите с тензором x ряд операций:
#– возведение в степень n, где n = 3, если ваш номер по списку группы в ЭИОС – четный и n = 2, если ваш номер по списку группы в ЭИОС – нечетный;
#– умножение на случайное значение в диапазоне от 1 до 10;
#– взятие экспоненты от полученного числа.

student_id =30
n=3
x_powered=x_float**n 
print(x_powered)
randon_number=randint(1,10)
x_multiplication=x_float*randon_number
print('умножение на случайное значение = ', randon_number)
print('Тензор', x_multiplication)
# взятие экспоненты от полученного числа
x_exp=torch.exp(x_multiplication)
print('взятие экспоненты от полученного числа.',x_exp)

#4 Вычислите и выведите на экран значение производной для полученного в п.3 значения по x.
#обратное распространение ошибки
x_exp.backward()

#градиент будем хранить в b
print(x_float.grad)

# На основе кода обучения линейного алгоритма создать код для решения задачи классификации цветков ириса из лабораторной работы №2.
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/student/Downloads/data.csv')
#df.colums = ['']
X = df.iloc[:,0:4].values

#  2 выходной тензор - значения, которые мы хотим предсказывать нашим алгоритмом
y = df.iloc[:,4].values

le =LabelEncoder()
y_encoder=le.fit_transform(y)

tensor_x = torch.tensor(X, dtype=torch.float32)
tensor_y = torch.tensor(y_encoder, dtype=torch.float32)
linear = nn.Linear(4, 1)
print ('w: ', linear.weight)
print ('b: ', linear.bias)
lossFn = nn.MSELoss() # MSE - среднеквадратичная ошибка, вычисляется как sqrt(sum(y^2 - yp^2))


# создадим оптимизатор - алгоритм, который корректирует веса наших сумматоров (нейронов)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.0001) # lr - скорость обучения

for i in range(0,900):
    pred = linear(tensor_x)
    loss = lossFn(pred, tensor_y)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    
    loss.backward()
    optimizer.step()

# посчитаем ошибки
sum_err = 0

for xi, target in zip(tensor_x, tensor_y):
    predict = linear(xi)
    if predict>0.5:
        predict=1
    else:
        predict=0
    sum_err += np.abs(target.detach().numpy() - predict)

print("Всего ошибок: ", sum_err)




























