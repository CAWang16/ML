import pandas as pd
import matplotlib.pyplot as plt

# read csv file
url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv'
data = pd.read_csv(url)

X = data['YearsExperience']
y = data['Salary']

# visualize data
plt.scatter(X , y, marker="x", color='lightcoral')
plt.title("年資-薪水")
plt.show()
