import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np


# read csv file
url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data2.csv'
data = pd.read_csv(url)

df = pd.DataFrame(data)
df.EducationLevel = df.EducationLevel.map({'高中以下':0,'大學':1,'碩士以上':2})

onehot_encoder = OneHotEncoder()
onehot_encoder.fit(data[['City']])
city_encoded = onehot_encoder.transform(data[['City']]).toarray()

df[['CityA','CityB','CityC']] = city_encoded
df = df.drop(['City','CityC'], axis=1)

X = df[['YearsExperience','EducationLevel','CityA','CityB']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()


w = np.array([1, 2, 2, 4])
b = 1


def compute_cost(X, y, w, b):
    y_pred = (X * w).sum(axis=1) + b  # 預測值
    cost = ((y - y_pred)**2 ).mean()
    return cost

# Optimizer: gradient descent - 根據斜率改變參數
# w - w 斜率 * learning_rate
def compute_gradient(X, y, w, b):
    y_pred = (X * w).sum(axis=1) + b
    b_gradient = (y_pred - y).mean()
    w_gradient = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        w_gradient[i] = (X[:,i]*(y_pred - y)).mean()

    return w_gradient, b_gradient

# 更新斜率
learning_rate = 0.001
w_gradient, b_gradient = compute_gradient(X_train, y_train, w, b)
# print(compute_cost(X_train, y_train, w, b))
w = w - w_gradient * learning_rate
b = b - b_gradient * learning_rate
# print(compute_cost(X_train, y_train, w, b))

c_hist = []
w_hist = []
b_hist = []

np.set_printoptions(formatter={'float': '{: .2e}'.format})
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

        if i % p_iter ==0 :
            print(f"Ieteration {i}: Cost: {cost:.2f}, w: {w}, b: {b:.2f}")

    return w, b, w_hist, b_hist, c_hist

w_init = np.array([1, 2, 2, 4])
b_init = 0
learning_rate = 1.0e-2
run_iter = 10000

w_final, b_final , w_hist, b_hist, c_hist = gradient_descent(X_train, y_train, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter)

y_pred = (w_final * X_test).sum(axis=1) + b_final
print(pd.DataFrame({
    'y_pred': y_pred,
    'y_test': y_test    
    })
)