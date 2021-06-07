import numpy as np

# 隐马尔科夫模型lambda=(A,B,pi)

A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
pi = [0.2, 0.3, 0.5]
T = 8
N = 3


O = ['红', '白', '红', '红', '白', '红', '白', '白']
o = np.zeros(T, int)              # 初始化为int型0数组
for i in range(T):
    if O[i] == '白':
        o[i] = 1     # 1代表白色
    else:
        o[i] = 0     # 0代表红色


# 动态规划求最优路径

pi_part = np.zeros((T, N))                  # 变量局部概率
reversal_analogy = np.zeros((T, N), int)    # 反向指针
a = []
best_path = np.zeros(T, int)


for i in range(N):
    pi_part[0][i] = pi[i] * B[i][o[0]]    # 初始化局部概率
    reversal_analogy[0][i] = 0

# 递推

for t in range(T-1):    #t=2,3,4....(T)
    t = t + 1
    for i in range(N):
        for j in range(N):
            a.append(pi_part[t - 1][j] * A[j][i])
        pi_part[t][i] = np.max(a) * B[i][o[t]]          # delta=max[delta*aji]*b(ot+1)
        reversal_analogy[t][i] = np.argmax(a, axis=0)   # psi=argmax[delta(t-1)aji]
        a = []
reversal_analogy = reversal_analogy + 1

# 终止

Pi_best_path = np.max(pi_part[T - 1])
best_path[T - 1] = np.argmax(pi_part[T - 1], axis=0) + 1



for t in range(T-1):       # 求最优路径回溯
    t = T - t - 2          # t=T-1,T-2,...,1
    a = t + 1
    b = best_path[t + 1] - 1
    best_path[t] = reversal_analogy[a][b]

print('最优路径为： ',best_path)
print('最优路径的概率为：',Pi_best_path)

