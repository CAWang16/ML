# Machine Learning Practice Files

This repository contains practice files for Machine Learning, focusing on different algorithms and techniques. Below is a description of the available files and their functionalities.

## Files

### 1. Simple_linear_regression.py

This script demonstrates how to perform simple linear regression and visualize the results. It includes a function to draw a predicted line on a graph based on the input parameters.

- **plot_pred**: This function takes two parameters, `w` for the slope and `b` for the intercept. It uses these parameters to calculate the predicted values and plots the predicted line on a scatter plot of the actual data points. This helps in visualizing the relationship between years of experience and salary.

### 2. Gradient_descent.py

This script demonstrates the implementation of gradient descent for linear regression. It includes functions to compute cost, compute gradients, and perform gradient descent optimization.

#### Description

This script performs simple linear regression using gradient descent. It reads a dataset of years of experience and corresponding salaries, then optimizes the linear regression model parameters (weights and bias) to minimize the prediction error.

#### Key Functions:

1. **compute_cost**: Calculates the mean squared error between the predicted and actual values.
2. **compute_gradient**: Computes the gradient of the cost function with respect to the weights (`w`) and bias (`b`).
3. **gradient_descent**: Iteratively updates the weights and bias to minimize the cost function using the computed gradients.

### 3. Multiple_linear_regression.py

This script demonstrates the implementation of multiple linear regression using gradient descent. It includes functions to compute cost, compute gradients, and perform gradient descent optimization on a dataset that includes categorical and numerical features.

#### Description

This script performs multiple linear regression using gradient descent. It reads a dataset containing years of experience, education level, city, and corresponding salaries. The script encodes categorical features, splits the data into training and testing sets, and then optimizes the linear regression model parameters (weights and bias) to minimize the prediction error.

#### Key Functions:

1. **compute_cost**: Calculates the mean squared error between the predicted and actual values.
2. **compute_gradient**: Computes the gradient of the cost function with respect to the weights (`w`) and bias (`b`).
3. **gradient_descent**: Iteratively updates the weights and bias to minimize the cost function using the computed gradients.

# Machine Learning Practice Files

這個儲存庫包含機器學習的練習檔案，專注於不同的算法和技術。以下是可用檔案及其功能的描述。

## Files

### 1. Simple_linear_regression.py

這個腳本展示了如何執行簡單線性回歸並視覺化結果。它包含了一個函數，用於在圖上繪製基於輸入參數的預測線。

- **plot_pred**: 這個函數接受兩個參數，`w` 代表斜率，`b` 代表截距。它使用這些參數來計算預測值，並在實際數據點的散點圖上繪製預測線。這有助於視覺化工作年限和薪水之間的關係。

### 2. Gradient_descent.py

這個腳本展示了如何實現線性回歸的梯度下降法。它包含計算成本、計算梯度以及執行梯度下降優化的函數。

#### Description

這個腳本使用梯度下降法執行簡單線性回歸。它讀取一個包含工作年限和相應薪水的數據集，然後優化線性回歸模型參數（權重和偏差），以最小化預測誤差。

#### Key Functions:

1. **compute_cost**: 計算預測值和實際值之間的均方誤差。
2. **compute_gradient**: 計算成本函數對權重（`w`）和偏差（`b`）的梯度。
3. **gradient_descent**: 迭代更新權重和偏差，使用計算出的梯度來最小化成本函數。

### 3. Multiple_linear_regression.py

這個腳本展示了如何使用梯度下降法實現多元線性回歸。它包含計算成本、計算梯度以及在包含類別和數值特徵的數據集上執行梯度下降優化的函數。

#### Description

這個腳本使用梯度下降法執行多元線性回歸。它讀取一個包含工作年限、教育水平、城市和相應薪水的數據集。腳本對類別特徵進行編碼，將數據分成訓練集和測試集，然後優化線性回歸模型參數（權重和偏差），以最小化預測誤差。

#### Key Functions:

1. **compute_cost**: 計算預測值和實際值之間的均方誤差。
2. **compute_gradient**: 計算成本函數對權重（`w`）和偏差（`b`）的梯度。
3. **gradient_descent**: 迭代更新權重和偏差，使用計算出的梯度來最小化成本函數。
