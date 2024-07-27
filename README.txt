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
