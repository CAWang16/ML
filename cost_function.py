import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.font_manager import fontManager
import numpy as np

# add font
fontManager.addfont("ChineseFont.ttf")
mlp.rc('font', family='ChineseFont')

# read csv file
url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv'
data = pd.read_csv(url)

X = data['YearsExperience']
y = data['Salary']

def compute_cost(X , y, w, b):
    y_pred = w * X + b
    cost = (y - y_pred) ** 2
    cost = cost.sum() / len(cost)

    return cost

# b = 0, w = -100 ~ 100

costs = []
for w in range (-100,101):
    cost = compute_cost(X, y, w, 10)
    costs.append(cost)

# 用點來呈現
# plt.scatter(range(-100,101), costs)

# 用線表示
# plt.plot(range(-100,101), costs)
# plt.title("cost function b = 0 w = -100 ~ 100")
# plt.xlabel("w")
# plt.ylabel("Cost")
# plt.show()

# 繪製3D圖
ws = np.arange(-100, 101)
bs = np.arange(-100, 101)
costs = np.zeros((201, 201))
i = 0
for w in ws:
    j = 0
    for b in bs:
        cost = compute_cost(X , y, w, b)
        costs[i,j] = cost
        j += 1
    i += 1

plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
pane_color = (254/255, 254/255, 254/255)
ax.xaxis.set_pane_color(pane_color)
ax.yaxis.set_pane_color(pane_color)
ax.zaxis.set_pane_color(pane_color)

b_grid, w_grod = np.meshgrid(bs, ws)
ax.plot_surface(w_grod, b_grid, costs, cmap='Spectral_r', alpha=0.7)
ax.plot_wireframe(w_grod, b_grid, costs, color='black', alpha=0.1)

ax.set_title("w b 對應的 cost")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_ylabel("Cost")



### 找出最低點
w_index, b_index = np.where( costs == np.min(costs))
print(w_index, b_index)
print(f"當w = {ws[w_index]} b = {bs[b_index]} 會有最小的cost:{costs[w_index, b_index]}")

ax.scatter(ws[w_index], bs[b_index], costs[w_index, b_index], color='red' , s=40)
plt.show()

