# (y - y_pred)**2
# (y - ( w * x + b)) **2
# w 微分 > 2x( w*x + b - y)
# b 微分 > 2(w*x + b -y)
# w - w方向斜率 * 學習率
# b - b方向斜率 * 學習率

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
from matplotlib.font_manager import fontManager
from ipywidgets import interact

# read csv file
url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv'
data = pd.read_csv(url)

X = data['YearsExperience']
y = data['Salary']

# load chinese font
# add font
fontManager.addfont("ChineseFont.ttf")
mlp.rc('font', family='ChineseFont')

def compute_cost(X , y, w, b):
    y_pred = w * X + b
    cost = (y - y_pred) ** 2
    cost = cost.sum() / len(cost)

    return cost

def compute_gradient(X, y, w, b):
    # w_gradient = (2* X * ( w*X + b - y)).mean()
    # b_gradient = (2 * ( w*X + b - y)).mean()
    w_gradient = (X * ( w*X + b - y)).mean()
    b_gradient = (( w*X + b - y)).mean()
    return w_gradient, b_gradient

w = 0
b = 0
learning_rate = 0.001
c_hist = []
w_hist = []
b_hist = []

def gradient_descent(X, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter=1000):
    w = w_init
    b = b_init
    for i in range(run_iter):

        
        w_gradient, b_gradient = gradient_function(X,y,w,b)
        w = w - w_gradient * learning_rate
        b = b - b_gradient * learning_rate
        cost = cost_function(X,y,w,b)
        
        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)

        # if i % p_iter ==0 :
        #     print(f"Ieteration {i}: Cost: {cost:.2f}, w: {w:.2f}, b: {b:.2f}")

    return w, b, w_hist, b_hist, c_hist
    
w_init = -100
b_init = -100
learning_rate = 1.0e-3
run_iter = 20000

w_final, b_final , w_hist, b_hist, c_hist = gradient_descent(X, y, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter)

plt.plot(np.arange(0, 100), c_hist[:100])
plt.title("Iteration vs Cost")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# 最後回到問題上 y = w*x + b
# print(f"年資3.5 預測薪資為 {w_final *3.5 + b_final:.1f}k")
# print(f"年資7 預測薪資為 {w_final *7 + b_final:.1f}k")


### 繪製3D圖

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
ax.plot_surface(w_grod, b_grid, costs, alpha=0.3)

ax.set_title("w b 對應的 cost")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_ylabel("Cost")



### 找出最低點
w_index, b_index = np.where( costs == np.min(costs))
print(w_index, b_index)
print(f"當w = {ws[w_index]} b = {bs[b_index]} 會有最小的cost:{costs[w_index, b_index]}")

ax.scatter(ws[w_index], bs[b_index], costs[w_index, b_index], color='red' , s=40)
ax.scatter(w_hist[0], b_hist[0], c_hist[0])
ax.plot(w_hist, b_hist, c_hist)

plt.show()
