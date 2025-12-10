#Решите задачу предсказания дохода по возрасту. 30 по списку в ЭИОС
import torch
import torch.nn as nn
import pandas as pd

df = pd.read_csv('dataset_simple.csv')
# возраст признак
X = torch.Tensor(df[['age']].values) 
# доход  целевая переменная    
y = torch.Tensor(df['income'].values)      

#создаем сеть для регрессии
class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size))

    def forward(self, X):
        return self.layers(X)

# рзмеры сети
inputSize = 1      
hiddenSize = 5     
outputSize = 1     

net = NNet_regression(inputSize, hiddenSize, outputSize)

# ф-ция ошибки и оптимизатор
lossFn = nn.L1Loss()                     
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

# обучение
epochs = 2000
for i in range(epochs):
    pred = net(X).squeeze()
    loss = lossFn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 200 == 0:
        print(f'эпоха {i}, ошибка- {loss.item()}')
        

# предсказание
with torch.no_grad():
    test_age = torch.Tensor([[40]])
    predicted_income = net(test_age)

print("предсказанный доход для возраста 40:")
print(predicted_income.item())