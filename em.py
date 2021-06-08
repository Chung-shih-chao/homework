import numpy as np
from numpy import *
import random
import copy

error = 0.00001  # 误差
# 用两个高斯分布生成数据

# 方差
sigma1 = 7
sigma2 = 3
# 均值
Miu1 = 20
Miu2 = 40
N = 100    # 采样点为100
X = mat(zeros((N, 1)))
for i in range(N):
    temp = random.uniform(0, 1)
    if (temp > 0.5):
        X[i] = temp * sigma1 + Miu1
    else:
        X[i] = temp * sigma2 + Miu2
# EM
k = 2
N = len(X)
Miu = np.random.rand(k, 1)
post_pi = mat(zeros((N, 2)))  # 后验概率
Q1_sigma = 0
Q1xi_sigma = 0
# 先求后验概率
for iter in range(100):
    for i in range(N):
        Q1_sigma = 0
        for j in range(k):
            Q1_sigma = Q1_sigma + np.exp(-1.0 / (2.0 * sigma1 ** 2) * (X[i] - Miu[j]) ** 2)
        for j in range(k):
            Q1xi_sigma = np.exp(-1.0 / (2.0 * sigma1 ** 2) * (X[i] - Miu[j]) ** 2)
            post_pi[i, j] = Q1xi_sigma / Q1_sigma
    oldMiu = copy.deepcopy(Miu)   # 保存miu
    # 最大化
    for j in range(k):
        Q1xi_sigma = 0
        Q1_sigma = 0
        for i in range(N):
            Q1xi_sigma = Q1xi_sigma + post_pi[i, j] * X[i]
            Q1_sigma = Q1_sigma + post_pi[i, j]
        Miu[j] = Q1xi_sigma / Q1_sigma
        # miu = Q1(z)(x1)+Q2(z)(x2)+.../Q1(z)+Q2(z)+...
    print(abs(Miu - oldMiu).sum())
    if (abs(Miu - oldMiu)).sum() < error:
        print(Miu)
        print(iter)
        break


