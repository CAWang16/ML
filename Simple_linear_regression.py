import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.font_manager import fontManager
from ipywidgets import interact

# read csv file
url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv'
data = pd.read_csv(url)

X = data['YearsExperience']
y = data['Salary']

# add font
fontManager.addfont("ChineseFont.ttf")
mlp.rc('font', family='ChineseFont')

# visualize data
# plt.scatter(X , y, marker="x", color='lightcoral')
# plt.title("年資-薪水")
# plt.xlabel("年資")
# plt.ylabel("月薪(千)")
# plt.show()

# y = wx + b

def plot_pred(w, b):
    y_pred = X * w + b

    plt.xlim([0,12])
    plt.ylim([-60,140])
    plt.plot(X, y_pred, label='預測縣')
    plt.scatter(X , y, marker="x", color='lightcoral', label='真實數據')
    plt.title("年資-薪水")
    plt.xlabel("年資")
    plt.ylabel("月薪(千)")
    plt.legend()
    plt.show()


plot_pred(20, 48)