import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Classification/Diabetes_Data.csv'
data = pd.read_csv(url)

data['Gender'] = data['Gender'].map({"男生":1,"女生":0})


X = data[['Age','Weight','BloodSugar','Gender']]
y = data['Diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 87)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

w = np.array([1,2,2,3])
b = 5
z = X_train @ w + b

def sigmoid(z):
    return 1/(1+np.exp(-z))
y_pred = sigmoid(z)

# -y * log(y_pred) - (1 - y) * log (1-y_pred)
# 設定 Cost function
# if y = 1 -> -log(y_pred) || y = 0 -> -log(1-y_ pred)


def compute_cost(X, y, w, b):
    z = X @ w + b
    y_pred = sigmoid(z)
    cost = -y*np.log(y_pred) - (1-y)*np.log(1-y_pred)
    cost = cost.mean()
    return cost

def compute_gradient(X, y, w, b):
    z = X @ w + b
    y_pred = sigmoid(z)
    b_gradient = (y_pred - y_train).mean()
    w_gradient = (X.T @ (y_pred - y)) / X.shape[0]
    return w_gradient, b_gradient
# print(compute_cost(X_train, y_train, w, b))

# np.set_printoptions(formatter={'float': '{:.2e}'.format})
def gradient_descent(X, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter=1000):
    w_hist = []
    b_hist = []
    c_hist = []

    w = w_init
    b = b_init
    for i in range(run_iter):
        w_gradient, b_gradient = gradient_function(X, y, w, b)
        w = w - w_gradient * learning_rate
        b = b - b_gradient * learning_rate
        cost = cost_function(X, y, w, b)

        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)

        if i % p_iter == 0:
            print(f"Iteration {i:5} : Cost {cost: .4e}, w: {w}, b: {b: .2e}, w_gradient:{w_gradient}, b_gradient:{b_gradient: .2e}")
    return w, b, w_hist, b_hist, c_hist
# print(compute_cost(X_train, y_train, w, b))


w_init = np.array([1,2,2,3])
b_init = 5
learning_rate = 1
run_iter = 2000
w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(X_train, y_train, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter)

print(w_final, b_final)

z = X_test @ w_final + b_final
y_pred = sigmoid(z)
# 把預測結果超過 0.5設置為(1)有糖尿病，反之，低於0.5設置為 0(無糖尿病)
y_pred = np.where(y_pred > 0.5 , 1 , 0)
# 計算正確率
accuracy = (y_pred == y_test).sum() / len(y_test) * 100
print(f"正確率為: {accuracy}")


# 假設有一真實情況 年齡:72 體重:92 血糖:102 女生
# 假設有二真實情況 年齡:65 體重:88 血糖:122 男生
X_real = np.array([[72, 92, 102, 0],[65,88,122,1]])
X_real = scaler.transform(X_real)
z = X_real @ w_final + b_final
y_real = sigmoid(z) * 100
print(f"例1患有糖尿病的機率為: {y_real[0]:.2f}%\n例2患有糖尿病的機率為: {y_real[1]:.2f}%")




### 視覺化查看計算過程
# plt.plot(np.arange(0, run_iter), c_hist[:run_iter], label= "Cost")
# plt.plot(np.arange(0, run_iter), w_hist[:run_iter], label= "w")
# plt.plot(np.arange(0, run_iter), b_hist[:run_iter], label= "b")
# plt.title("Iteration vs Cost")
# plt.xlabel("Iteration")
# plt.ylabel("cost")
# plt.legend()
# plt.show()