# Machine Learning Algorithms from Scratch

This repository contains three Python scripts that implement machine learning algorithms from scratch without relying on `nn.Module`. This approach allows for a more detailed exploration of each computational step, helping to build a deeper understanding of how gradient descent, cost functions, and other core ML concepts work.

## 1. **Multiple Linear Regression (multiple_linear_regression.py)**

This script implements a multiple linear regression model from scratch to predict salaries based on features such as years of experience, education level, and city.

### Key Features:

- **Data Preprocessing**: Uses `OneHotEncoder` for categorical feature encoding and `StandardScaler` to standardize numerical features.
- **Gradient Descent**: Manually computes the gradients for weights and bias using mean squared error (MSE) as the cost function.
- **Model Training**: Performs gradient descent over multiple iterations to minimize the cost and improve the model.
- **Prediction**: Predicts salary for candidates based on their features, such as experience and education level.

### Key Libraries:

- `numpy`, `pandas`, `scikit-learn` (for preprocessing)
- **Note**: This script does not use `nn.Module` to manually explain all calculations in detail.

---

## 2. **Logistic Regression Classification (classification_logistic_regression.py)**

This script implements logistic regression from scratch for classifying diabetes cases based on features like age, weight, blood sugar, and gender.

### Key Features:

- **Logistic Function (Sigmoid)**: Implements the logistic function to map predictions to probabilities.
- **Gradient Descent**: Manually calculates gradients to optimize weights and bias using binary cross-entropy as the cost function.
- **Prediction**: Classifies patients as diabetic or non-diabetic, and calculates model accuracy.

### Key Libraries:

- `numpy`, `pandas`, `scikit-learn`
- **Note**: All gradient and cost computations are manually implemented to avoid using higher-level abstractions like `nn.Module`.

---

## 3. **Gradient Descent Visualization (gradient_descent.py)**

This script provides a visual exploration of gradient descent for linear regression, showing how weights (`w`) and bias (`b`) are updated over time.

### Key Features:

- **Cost Function Visualization**: Shows the cost surface and the movement of weights and bias across iterations.
- **3D Plot**: Visualizes the cost in relation to the weights and bias, highlighting the minimum cost point.
- **Convergence Plot**: Plots the cost reduction over iterations for a clearer understanding of how gradient descent works.

### Key Libraries:

- `matplotlib`, `numpy`, `pandas`

---

### How to Run the Scripts:

1. Clone this repository and navigate to the script you wish to run.
2. Ensure all required libraries are installed:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
3. Run the Python script:
   ```bash
   python multiple_linear_regression.py
   python classification_logistic_regression.py
   python gradient_descent.py

# 從零實現機器學習算法

本倉庫包含三個 Python 腳本，這些腳本從零實現了機器學習算法，不依賴於 `nn.Module`。這種方法允許更詳細地探索每個計算步驟，有助於深入理解梯度下降、損失函數和其他核心機器學習概念的工作原理。

## 1. **多元線性回歸 (multiple_linear_regression.py)**

該腳本從零實現了一個多元線性回歸模型，用於根據工作年限、學歷和城市等特徵來預測薪資。

### 主要功能：

- **數據預處理**: 使用 `OneHotEncoder` 對分類特徵進行編碼，並使用 `StandardScaler` 對數值特徵進行標準化。
- **梯度下降**: 手動計算權重和偏差的梯度，使用均方誤差 (MSE) 作為損失函數。
- **模型訓練**: 通過多次迭代的梯度下降來最小化成本並提高模型性能。
- **預測**: 根據工作經驗和學歷等特徵預測候選人的薪資。

### 主要庫：

- `numpy`, `pandas`, `scikit-learn`（用於數據預處理）
- **注意**: 該腳本不使用 `nn.Module`，以便更詳細地解析所有計算步驟。

---

## 2. **邏輯回歸分類 (classification_logistic_regression.py)**

該腳本從零實現了一個邏輯回歸模型，用於根據年齡、體重、血糖和性別等特徵分類糖尿病患者。

### 主要功能：

- **邏輯函數 (Sigmoid)**: 實現邏輯函數，將預測映射為概率。
- **梯度下降**: 手動計算梯度，通過二元交叉熵作為損失函數來優化權重和偏差。
- **預測**: 將患者分類為有糖尿病或無糖尿病，並計算模型的準確率。

### 主要庫：

- `numpy`, `pandas`, `scikit-learn`
- **注意**: 所有梯度和損失計算均手動實現，以避免使用像 `nn.Module` 這樣的高級抽象。

---

## 3. **梯度下降可視化 (gradient_descent.py)**

該腳本提供了一個梯度下降的可視化探索，用於線性回歸，展示權重 (`w`) 和偏差 (`b`) 如何隨時間更新。

### 主要功能：

- **損失函數可視化**: 顯示損失曲面以及權重和偏差在迭代過程中的變動。
- **3D 圖**: 將權重和偏差與損失的關係可視化，突出顯示最低成本點。
- **收斂圖**: 繪製成本隨迭代減少的曲線，以便更清楚地了解梯度下降的工作原理。

### 主要庫：

- `matplotlib`, `numpy`, `pandas`

---

### 如何運行腳本：

1. 克隆此倉庫並進入要運行的腳本目錄。
2. 確保安裝了所有所需的庫：
   ```bash
   pip install numpy pandas scikit-learn matplotlib

3. 運行 Python 腳本：
   ```bash
   python multiple_linear_regression.py
   python classification_logistic_regression.py
   python gradient_descent.py

---

### 數據集來源：

- **薪資數據集 (Salary Data)**:
  - 此數據集用於 `multiple_linear_regression.py` 腳本中的線性回歸模型，用於預測不同特徵下的薪資水平。
  - [Salary Dataset](https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data2.csv)

- **糖尿病數據集 (Diabetes Data)**:
  - 此數據集用於 `classification_logistic_regression.py` 腳本中的邏輯回歸模型，用於分類糖尿病患者。
  - [Diabetes Dataset](https://raw.githubusercontent.com/GrandmaCan/ML/main/Classification/Diabetes_Data.csv)

---

### 注意事項：

- **不使用 nn.Module**: 為了詳細解析機器學習算法的每一步，我們在這些腳本中未使用 `nn.Module`。所有的梯度計算、損失函數和模型優化都是手動實現的，這樣可以幫助理解核心機器學習概念。
- **可視化支持**: `gradient_descent.py` 提供了可視化的支持，能夠更直觀地查看梯度下降過程以及權重、偏差如何隨著每次迭代而更新。

---

### 未來改進：

- **模型擴展**: 可以將這些基本實現擴展到更複雜的模型，如多層感知器 (MLP) 和卷積神經網絡 (CNN)，以進一步探索深度學習。
- **數據增強**: 未來可以增加更多的數據預處理步驟和增強技術，以提高模型的泛化能力。

---

### 貢獻方式：

歡迎提交 PR 來改進此項目，或者提供任何意見反饋。您可以通過創建 Issues 或直接 Fork 本倉庫進行貢獻。
