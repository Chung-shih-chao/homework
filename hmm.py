import numpy as np

# 前向算法
def forward_al(A, B, p_i, o, T, N):

    # 输入为隐马尔科夫模型观测序列O
    # 输出观测序列发生的条件概率

    alpha = np.zeros((T, N))
    for i in range(N):    # 计算初值
        h = o[0]
        alpha[0][i] = p_i[i] * B[i][h]  # 公式：a(i)=p(i)b(o)

    for t in range(T-1):   # 递推
        h = o[t+1]
        for i in range(N):
            a = 0
            for j in range(N):
                a += (alpha[t][j] * A[j][i])
            alpha[t+1][i] = a * B[i][h]     # at+1(i) = [a(1)a1i+...+at(j)aji+....]b(ot+1),i=1,2,....MN
    P = 0
    for i in range(N):
        P += alpha[T-1][i]   # 终止 P(O|z)=aT(1)+aT(2)+...aT(n)...
    return P, alpha

def back_al(A, B, p_i, o, T, N):

    # 设置初值

    beta = np.ones((T, N))

    # 递推

    for t in range(T-1):    # t=T-1, T-2,....
        t = T - t - 2
        h = o[t + 1]
        h = int(h)

        for i in range(N):  # 公式：beta(i)=aijbj(ot+1)+1(j)求和
            beta[t][i] = 0  # 从后往前推，beta=A*B*o*beta=N*N*N*M*M*N*N*1=N*1
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][h] * beta[t+1][j]
    # 终止

    P = 0
    for i in range(N):  # 公式：P(o|lambda)=pai*bi(o1)*beta1(i)
        h = o[0]
        h = int(h)
        P += p_i[i] * B[i][h] * beta[0][i]
    return P, beta



T = 8
N = 3
# 马尔科夫模型lambda=(A,B,pai)
A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
pi = [0.2, 0.3, 0.5]
O = ['红', '白', '红', '红', '白', '红', '白', '白']
o = np.zeros(T, int)
for i in range(T):
    if O[i] == '白':
        o[i] = 1
    else:
        o[i] = 0
PF, alpha = forward_al(A, B, pi, o, T, N)
PB, beta = back_al(A, B, pi, o, T, N)
print("PF:", PF, "PB:", PB)
P = alpha[4 - 1][3 - 1] * beta[4 - 1][3 - 1]
print("P(i4=q3|O,lambda)=", P / PF)
