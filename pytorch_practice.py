#%%
import pandas as pd
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
import torch
# import matplotlib.pyplot as plt
# import matplotlib as mlp
# from matplotlib.font_manager import fontManager
# from ipywidgets import interact

# read csv file
url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv'
data = pd.read_csv(url)

X = data['YearsExperience']
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)
# transform to numpy before transform to tensor
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
#transform to tensor
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.rand(1, requires_grad=True))
        self.b = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, x):
        return self.w*x + self.b

torch.manual_seed(87)
model = LinearRegressionModel()

#%%

cost_fn = nn.MSELoss()
y_pred = model(X_train)
cost = cost_fn(y_pred , y_train)

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
optimizer.zero_grad()
cost.backward()
optimizer.step()

print(model.state_dict())
print(cost)


# %%
for i in range (100):
    model.train()
    y_pred = model(X_train)
    cost = cost_fn(y_pred, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_cost = cost_fn(test_pred, y_test)

    if i % 10 == 0:
        print(f'i: {i:5} cost: {cost: .4e} test_cost: {test_cost: .4e}')
# %%
