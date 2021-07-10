import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 获得数据
li = load_iris()

# 处理数据


def create_iris_data():
    """加载iris鸢尾花数据集"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:150, [0, 1, 2,3,-1]])
    print("已加载iris鸢尾花数据集")
    return data

# 2.取出目标值
data = create_iris_data()
x, y = data[:, :4], data[:, -1]


# 3.将数据拆分成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# 4.距离函数
def distance(data1,data2):
    res = 0
    for i in range(len(data1)):
        res += (data1[i]-data2[i])**2
    return res**0.5    # 欧几里得距离

# knn算法


def knn(x_train, x_test, y_train):

    x_iris1 = []  # 1类集合
    x_iris2 = []  # 2类集合
    x_iris3 = []  # 3类集合

    y_hat = []  # 预测值
    # print(y_train)
    for i in range(len(y_train)): # 通过标签，把训练集按种类分类
        if y_train[i] == 0:       # 如果是1类，加入x_iris1中
            x_iris1.append(x_train[i])
        if y_train[i] == 1:
            x_iris2.append(x_train[i])
        if y_train[i] == 2:
            x_iris3.append(x_train[i])


    for i in range(len(x_test)):
        x_temp = x_test[i]     # 当前检测的测试集中的一个样本
        res = [[], [], []]
        # 求该样本与训练集中各类别样本的距离之和
        for x_1 in x_iris1:
            res[0].append(distance(x_1, x_temp))

        for x_2 in x_iris2:
            res[1].append(distance(x_2, x_temp))

        for x_3 in x_iris3:
            res[2].append(distance(x_3, x_temp))

        # 每组res取最小k个
        K = 10
        res_k = [[], [], []]
        for j in range(3):
            res[j].sort()
            res_k[j] = res[j][0:K]
        res_k_sum = [
            sum(res_k[0]),
            sum(res_k[1]),
            sum(res_k[2]),
        ]
        min_res = min(res_k_sum) # 找出这三者平均距离最小的点对应的索引
        for j in range(3):
            if res_k_sum[j] is min_res:
                y_hat.append(j)
    return y_hat


def calc_acc(y, y_hat):
    """计算准确率
    """
    acc = 0
    if type(y_hat) != 'numpy.ndarray':
        y_hat = np.array(y_hat)
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            acc += 1/len(y)
    return acc


y_hat = knn(x_train, x_test, y_train)
acc = calc_acc(y_test, y_hat)
print(y_hat)
print(y_test)
print(acc)
