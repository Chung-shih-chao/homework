import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :2], data[:, -1]


class Adaboost:
    def __init__(self, x, y, lr=1.0):
        self.x = x
        self.y = y
        self.lr = lr                                                                             # 学习率
        self.classifiers = []                                                                    # 子分类器集合
        self.alphas = []                                                                         # 子分类器权值
        self.num_samples = len(self.x)                                                           # 样本个数
        self.Dn = np.array([1/self.num_samples] * self.num_samples)                              # 数据权重

    def addClassifier(self, classifier=DecisionTreeClassifier(max_depth=1)):

        classifier.fit(self.x, self.y, sample_weight=self.Dn)                                    # 训练子分类器
        y_pre = classifier.predict(self.x)                                                       # 子分类器预测
        em = np.sum((y_pre != self.y) * self.Dn) / np.sum(self.Dn)                               # 计算加权错误率
        alpha = 0.5 * self.lr * np.log((1 - em) / em)                                            # 计算alpha
        self.Dn = (self.Dn * np.exp(-alpha * y_pre * self.y)) / np.sum(self.Dn)                  # 更新权值
        self.classifiers.append(classifier)                                                      # 收集子分类器
        self.alphas.append(alpha)                                                                # 收集alpha

    def predict(self, x, original=False):                                                        # 强分类器输出
        y_pre = np.zeros([len(x)]).astype("float")
        for classifier, alpha in zip(self.classifiers, self.alphas):
            y_pre += alpha * classifier.predict(x)                                               # 构建基本分类器的线性组合
        if original:                                                                             # 是否原始输出
            return y_pre
        else:
            y_pre = np.sign(y_pre)                                                               # 最终分类器
            return y_pre

    def plot(self, style='2d'):
        if len(self.x.shape) != 2:
            return

        y_predict = self.predict(self.x)                                                         # 子分类器预测
        error_rate = np.sum(y_predict != self.y)/self.num_samples                                # 误差率

        fig = plt.figure(figsize=(5, 4), dpi=140)

        x_min, x_max = np.min(self.x[:, 0]-2, axis=0), np.max(self.x[:, 0]+2, axis=0)
        y_min, y_max = np.min(self.x[:, 1]-2, axis=0), np.max(self.x[:, 1]+2, axis=0)

        if style == "2d":
            test_X,test_Y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]                         # 生成网络采样点
            grid_test = np.stack((test_X.flat, test_Y.flat), axis=1)                             # 测试点
            grid_hat = self.predict(grid_test)                                                   # 预测分类值
            grid_hat = grid_hat.reshape(test_X.shape)

            ax = fig.add_subplot(1, 1, 1)
            ax.set(title='Adaboost_iris(N:{},em:{})'.format(len(self.alphas), error_rate))
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            cm_light = matplotlib.colors.ListedColormap(['#00FFFF', '#E50000'])                  # 配置颜色
            ax.pcolormesh(test_X, test_Y, grid_hat, cmap=cm_light)
            ax.scatter(self.x[self.y == -1][:, 0], self.x[self.y == -1][:, 1], marker='x')
            ax.scatter(self.x[self.y == 1][:, 0], self.x[self.y == 1][:, 1], marker='^')
        plt.show()


if __name__ == '__main__':
    x, y = create_data()
    model = Adaboost(x, y, lr=0.6)
    for i in range(50):
        model.addClassifier(classifier=DecisionTreeClassifier(max_depth=1))
    y_predict = model.predict(x)
    plt.show()
    model.plot(style='2d')
