import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Logistic_Regression:
    def __init__(self):
        self.w = None

    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def output(self, x):
        z = np.dot(self.w, x.T)
        a = self.sigmoid(z)
        return a

    def compute_loss(self, x, y):
        num_train = x.shape[0]   # 样本数量
        a = self.output(x)       # 输出
        loss = np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a)) / num_train # 所有样本的平均loss
        dw = np.dot((a - y), x) / num_train   # 平均梯度
        return loss, dw

    def train(self, x, y, lr=0.01, num_iterations=20000):
        num_train, num_features = x.shape    # 样本数、特征数（80,2）
        self.w = 0.001 * np.random.randn(1, num_features)
        loss = []
        for i in range(num_iterations):
            error, dw = self.compute_loss(x, y)
            loss.append(error)
            self.w -= lr * dw
            # if i % 200 == 0:
                # print("step:[%d/%d],loss:%f" % (i, num_iterations, error))
        return loss

    def predict(self, x):
        a = self.output(x)
        y_pred = np.where(a >= 0.5, 1, 0)

        return y_pred

class MulityLR:
    def __init__(self, X, y):
        self.features = X
        self.labels = y

        self.clfs = []                              # 储存训练的LR二分类器
        self.class_num = 3                          # 类别数量
        self.clf_num   = 3                          # 分类器数量
        self.clf_class = [[0, 1], [0, 2], [1, 2]]   # 分类器与所分各类的对应关系


    def train(self):
        for i in range(0, self.clf_num):
            # 找到所分的两类
            class_1 = self.clf_class[i][0]
            class_2 = self.clf_class[i][1]

            # 遍历数据集找出所分的两类，将他们的特征和标签加入feature和label
            feature = []    # 当前分类器所分两类的feature
            label = []      # 当前分类器所分两类的label
            for j in range(len(self.labels)):
                if self.labels[j] == class_1:
                    feature.append(self.features[j])
                    label.append(self.labels[j])
                if self.labels[j] == class_2:
                    feature.append(self.features[j])
                    label.append(self.labels[j])

            if class_1 == 0 and class_2 == 2:
                for i in range(len(label)):
                    if label[i] == 2.0:
                        label[i] = 1

            if class_1 == 1 and class_2 == 2:
                for i in range(len(label)):
                    if label[i] == 1:
                        label[i] = 0
                    if label[i] == 2:
                        label[i] = 1

            # 针对当前两类进行训练
            model = Logistic_Regression()   # 初始化当前LR模型
            feature, label = np.array(feature), np.array(label)
            model.train(feature, label)
            self.clfs.append(model)
        print()


    def predict(self, X):
        """预测
        :params X: 要进行预测的样本
        """
        # 此处输入的feature是待预测的feature
        pred = []  # pred的每个元素是当前分类器对所有实例的预测
        pred_result = []  # 存放最后预测结果（未对应回原label）

        for i in range(len(self.clfs)):
            model = self.clfs[i]
            # 用所有分类器进行预测，结果存入single_temp
            pred_temp = model.predict(X)
            pred_temp = pred_temp[0]
            if i == 1:
                for j in range(len(pred_temp)):
                    if pred_temp[j] == 1:
                        pred_temp[j] = 2
            if i == 2:
                for j in range(len(pred_temp)):
                    if pred_temp[j] == 0:
                        pred_temp[j] = 1
                        if(j < len(pred_temp)-1):
                            j += 1
                    if pred_temp[j] == 1:
                        pred_temp[j] = 2


            pred.append(pred_temp)


        for i in range(0, np.shape(X)[0]):  # 对每个实例的分类结果,i代表一个实例
            pred_single = []
            for j in range(3):  # 当前特征下每个二分类器的结果,j代表一个二分类器
                a = pred[j]
                c = a[i]
                pred_single.append(c)
            pred_result.append(self.find_most_elem(pred_single))

        return pred_result

    def find_most_elem(self, data):
        """寻找出现最多的元素
        """
        result = None  # 存放
        result_dict = Counter(data)  # 获取每个元素出现次数的字典
        for key, value in result_dict.items():
            if value == max(result_dict.values()):
                result = key
        return result


def divide_train_test(x, y):
    from sklearn.model_selection import train_test_split
    """划分测试集与训练集
    :param:     x features
    :param:     y labels
    :return:    x_train, y_train, x_test, y_test    <list>
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return x_train, y_train, x_test, y_test

def create_iris_data():
    """加载iris鸢尾花数据集"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:150, [0, 1, 2, 3, -1]])
    print("已加载iris鸢尾花数据集")
    return data

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


# 获得数据

# 2.取出目标值
data = create_iris_data()
x, y = data[:, :4], data[:, -1]
x_train, y_train, x_test, y_test = divide_train_test(x, y)

MLR = MulityLR(x_train, y_train)
MLR.train()
y_hat = MLR.predict(x_test)
acc = calc_acc(y_test,y_hat)
print(y_hat)
print(y_test)
print(acc)

