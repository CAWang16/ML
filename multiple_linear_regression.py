import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data2.csv'
data = pd.read_csv(url)

# 資料預先處理
data['EducationLevel'] =  data['EducationLevel'].map({'碩士以上':2,'大學':1,'高中以下':0})
encoder = OneHotEncoder()
encoder.fit(data[['City']])
encoded_column = encoder.transform(data[['City']]).toarray()

data[['CityA','CityB','CityC']] = encoded_column
data = data.drop(['City','CityC'], axis = 1)

X = data[['YearsExperience','EducationLevel','CityA','CityB']]
y = data['Salary']

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# 資料標準化
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)

w = np.zeros(X_train.shape[1])
b = 0

# 寫出計算 cost 的函式
def compute_cost(X, y, w, b):
    y_pred = (X @ w) + b
    cost = ((y - y_pred)**2).mean()
    return cost

# 計算 gradient 的函式 (y_pred, b_gradient, w_gradient)
def compute_gradient(X, y, w,b):
    y_pred = (X @ w) + b
    error = y_pred - y
    b_gradient = (y_pred - y).mean()
    w_gradient = (X.T @ error / X.shape[0])

    return w_gradient, b_gradient


# 設置字顯示方式
np.set_printoptions(formatter={'float': '{:.2e}'.format})

# 
def gradient_descent(X, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter=300):
    # 紀錄每次 w, b, cost的更新
    w_hist = []
    b_hist = []
    c_hist = []

    w = w_init
    b = b_init
    # 執行迴圈使梯度下降
    for i in range(run_iter):
        w_gradient, b_gradient = gradient_function(X, y, w, b)

        w = w - w_gradient * learning_rate
        b = b - b_gradient * learning_rate
        cost = cost_function(X, y, w, b)

        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)

        if i % p_iter == 0:
            print(f"Iteration {i}: Cost: {cost:.4e}, w: {w}, b: {b:.2e}")

    return w, b, w_hist, b_hist, c_hist


learning_rate = 1.0e-2
run_iter = 3000

w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(X_train, y_train, w, b, learning_rate, compute_cost, compute_gradient, run_iter)

# 查看訓練以及測試用的 cost
print("Train cost:",compute_cost(X_train, y_train, w_final, b_final))
print("Test cost:", compute_cost(X_test, y_test, w_final, b_final))

# 查看預測值 以及 測試組的預測結果
y_pred = (w_final * X_test).sum(axis=1) + b_final
print(pd.DataFrame({
    'y_pred': y_pred,
    'y_test': y_test    
    })
)

# 使用模型來預測薪資
### A 求職者: 年資7年(7), 碩士以上(2), CityA(1,0)
### B 求職者: 年資3年(3), 大學(1), CityC(0,0)
Candidates = np.array([[7,2,1,0],[3,1,0,0]])
Candidates = Scaler.transform(Candidates)
y_pred = Candidates @ w_final + b_final
print(f"候選人 A, B 預測薪資為: {y_pred[0]:.1f} 及 {y_pred[1]:.1f} k")



