import numpy as np
from math import *
from prop_network import adj_matrix
from prop_network import adj_matrix2
import matplotlib.pyplot as plt
import random
import pandas as pd

# def loc (a, b, sum_la_pulse, sum_lu_pulse, sum_mu_pulse):  ###网格搜索找到满足条件的a,b的下标
#
#     values = []
#     indexes = []
#     for m in range(len(a)):
#         val = []
#         ind = []
#         for n in range(m, len(b)):
#             y = a[m] + b[n] + sum_la_pulse + sum_lu_pulse - sum_mu_pulse
#             val.append(y)
#             ind.append((m, n))
#         minm1 = min(val)
#         ind1 = val.index(minm1)
#         values.append(minm1)
#         indexes.append(ind[ind1])
#     minm2 = min(values)
#     ind2 = values.index(minm2)
#     return indexes[ind2]  ##检查列表和数组的精度

# def policy(tauk, lambda_a, lambda_u, mu, sa, su, ca, cu):
#     for t in range(t_interval):
#         for tk in tauk:
#             if tk == t:
#                 sum_la_pulse = lambda_a[tk] * g_func(cg, ca[tk]) @ sa[tk]
#                 sum_lu_pulse = lambda_u[tk] * f_func(cg, cu[tk]) @ su[tk]
#                 sum_mu_pulse = mu[tk] * (g_func(cg, ca[tk]) @ sa[tk] + f_func(cg, cu[tk]) * su[tk])
#                 # =================search optimal policy======================
#                 aa = np.arange(a_low, a_upp, 0.1)  ### paremeter
#                 bb = np.arange(u_low, u_upp, 0.1)
#                 locs = loc(aa, bb, sum_la_pulse, sum_lu_pulse, sum_mu_pulse)
#                 m = locs[0]
#                 n = locs[1]
#                 ca[tk] = aa[m]
#                 cu[tk] = bb[n]
#                 print("ca=", ca[tk])
#                 print("cu=", cu[tk])
#     return ca, cu

# def profit(r, ca, cu):
#
#     sum_tmp = 0
#     sum_ca = 0
#     sum_cu = 0
#     for t in range(t_interval): ## [0, T/h]
#         tmp = r[t].sum(axis=0)
#         sum_tmp += tmp
#         for tk in tauk:
#             if tk==t:
#                sum_ca += ca[tk]
#                sum_cu += cu[tk]
#     prof = omega * sum_tmp * h - sum_ca - sum_cu
#     return prof

# def policy_sim(tauk, lambda_a, lambda_u, mu, sa, su, ca, cu):
#     for t in range(t_interval):
#         for tk in tauk:
#             if tk == t: # accommulate sum (a single value not a vector)
#                 sum_la_pulse = (lambda_a[tk] * g_func(cg, ca[tk])) @ sa[tk]
#                 sum_lu_pulse = (lambda_u[tk] * f_func(cg, cu[tk])) @ su[tk]
#                 sum_mu_pulse = sum(mu[tk] * (g_func(cg, ca[tk]) * sa[tk] + f_func(cg, cu[tk]) * su[tk]))
#                 print("sum_lu_pulse=", sum_lu_pulse)
#                 print("sum_la_pulse=", sum_la_pulse)
#                 print("sum_mu_pulse=", sum_mu_pulse)
#                 # =================search optimal policy======================
#                 aa = np.arange(a_low, a_upp, 0.1)  ### paremeter
#                 bb = np.arange(u_low, u_upp, 0.1)
#                 # loc(aa, bb, sum_la_pulse, sum_lu_pulse, sum_mu_pulse)
#                 locs = loc_sim(aa, bb, sum_la_pulse, sum_lu_pulse, sum_mu_pulse)
#                 m = locs[0]
#                 n = locs[1]
#                 ca[tk] = aa[m]
#                 cu[tk] = bb[n]
#                 print("ca=", ca[tk])
#                 print("cu=", cu[tk])
#     return ca, cu

# def pulse_time(thresh):
#     ## K>1,
#     ## output c list
#     D = K - 1
#     # num = len(str(h)) - 2
#     rest = []
#     for n in range(1, math.ceil(t_interval / D)):  ### 除过脉冲单位的剩下的所有可能的时刻
#         rst = t_interval - D * n
#         if rst >= thresh:
#             rest.append(rst)
#     rest.sort()  ## 升序排列
#     # print(rest)
#     c = []
#     for term in range(len(rest)):
#         b = []
#         if (rest[term] % 2 == 0):  ### 判断是否为偶数
#             a1 = rest[term] / 2
#             b.append(int(a1))
#             for m in range(K - 1):
#                 a1 += (t_interval - rest[term]) / (K - 1)  ## 每个脉冲单位（第一时刻+脉冲单位）
#                 b.append(int(a1))  ##  保证两边的下标都是整形
#         else:
#             a1 = (rest[term] - 1) / 2
#             b.append(int(a1))
#             for m in range(K - 1 ):
#                 a1 += ((t_interval - 1) - (rest[term] - 1)) / (K - 1)
#                 b.append(int(a1))
#         c.append(b)
#     return c
# def f_func(cf, x):
#     return cf *


# def f_func(cf, x):
#     return cf * (exp(0.5*x)-1)/(exp(x)+1)
# #0.6*x/(x+5)
# #(exp(0.5*x)-1)/(exp(x)+1)
# ## 0.3/pi * np.arctan(0.5*x)
# # 0.03*sqrt(x)
# def g_func(cg, x):
#     return cg * 0.03*sqrt(x)

def f_func(cf, x):
    return cf * 0.5 /pi * np.arctan(0.5*x)
def g_func(cg, x):
    return cg * 0.6 *x/(x+5)
#
#
def sys(sa_t, su_t, r_t, sum_sa, sum_su, sum_r, beta1, beta2, eta, delta, gamma_a, gamma_u):  # states of the node i at time t

    dsa_t = beta1 * (1 - sa_t - su_t - r_t) * sum_sa + eta * su_t * sum_sa - delta * sa_t - gamma_a * sa_t * sum_r

    dsu_t = beta2 * (1 - sa_t - su_t - r_t) * sum_su - eta * su_t * sum_sa - delta * su_t - gamma_u * su_t * sum_r

    dr_t = gamma_u * su_t * sum_r + gamma_a * sa_t * sum_r - delta * r_t

    return dsa_t, dsu_t, dr_t

def adj(lambda_at, lambda_ut, mu_t, sum_sa, sum_su, sum_r, sum_sasur_la, sum_su_la, sum_su_lu, sum_sasur_lu, sum_sa_la,sum_su_mu, sum_sa_mu, beta1, beta2, eta, delta, gamma_a, gamma_u,omega):

    dlambda_at = (beta1 * sum_sa + gamma_a * sum_r + delta) * lambda_at - (beta1 * sum_sasur_la + eta * sum_su_la) + beta2 * sum_su * lambda_ut + eta * sum_su_lu - gamma_a * sum_r * mu_t

    dlambda_ut = (beta1 * sum_sa - eta * sum_sa) * lambda_at + (beta2 * sum_su + gamma_u * sum_r + eta * sum_sa + delta) * lambda_ut - beta2 * sum_sasur_lu - gamma_u * sum_r * mu_t

    dmu_t = - omega + beta1 * sum_sa * lambda_at + gamma_a * sum_sa_la + beta2 * sum_su * lambda_ut + gamma_u * sum_su_lu + delta * mu_t - (gamma_u * sum_su_mu + gamma_a * sum_sa_mu)

    return dlambda_at, dlambda_ut, dmu_t

# def forward (tauk, sa, su, r, ca, cu):
def forward(sa, su, r, ca, cu, t_interval, pulse_interval, a, beta1, beta2, eta, delta, gamma_a, gamma_u,h, g_func, f_func, cg, cf):
    for t in range(t_interval-1):  # [0, T/h-1]
            if t!=0 and t % pulse_interval==0:
                # print("pulse forward t:",t)
                sa[t] = sa[t] - g_func(cg, ca[t]) * sa[t]
                su[t] = su[t] - f_func(cf, cu[t]) * su[t]
                r[t] = r[t] + g_func(cg, ca[t]) * sa[t] + f_func(cf, cu[t]) * su[t]
                #=============nature evolution=================
                sum_sa = np.dot(a, sa[t])
                sum_su = np.dot(a, su[t])
                sum_r = np.dot(a, r[t])
                dsa, dsu, dr = sys(sa[t], su[t], r[t], sum_sa, sum_su, sum_r, beta1, beta2, eta, delta, gamma_a, gamma_u)
                sa[t + 1] = sa[t] + h * dsa
                su[t + 1] = su[t] + h * dsu
                r[t + 1] = r[t] + h * dr

            else:
                # print("nonpulse forward t:", t)
                sum_sa = np.dot(a, sa[t])
                sum_su = np.dot(a, su[t])
                sum_r = np.dot(a, r[t])
                ### 向前更新三个状态变量
                dsa, dsu, dr = sys(sa[t], su[t], r[t], sum_sa, sum_su, sum_r, beta1, beta2, eta, delta, gamma_a, gamma_u)
                sa[t + 1] = sa[t] + h * dsa
                su[t + 1] = su[t] + h * dsu
                r[t + 1] = r[t] + h * dr

    return sa, su, r


def backward(lambda_a, lambda_u, mu, ca, cu, sa, su, r, t_interval, pulse_interval, a, beta1, beta2, eta, delta, gamma_a, gamma_u, omega, h, cg, cf):
    for t in range(t_interval-1, 0, -1):  # T->1

        if t != t_interval - 1 and t % pulse_interval == 0:
            # print("pulse forward t:", t)
            lambda_a[t] = lambda_a[t] - lambda_a[t] * g_func(cg, ca[t]) + mu[t] * g_func(cg, ca[t])
            lambda_u[t] = lambda_u[t] - lambda_u[t] * f_func(cf, cu[t]) + mu[t] * f_func(cf, cu[t])
            # =========================== nature evolution============
            sum_sa = np.dot(a, sa[t])
            sum_su = np.dot(a, su[t])
            sum_r = np.dot(a, r[t])
            sum_sasur_la = a @ ((1 - sa[t] - su[t] - r[t]) * lambda_a[t])
            sum_su_la = a @ (su[t] * lambda_a[t])
            # print("sum_su_la=", sum_su_la)
            sum_su_lu = a @ (su[t] * lambda_u[t])
            # print("sum_su_lu=",sum_su_lu)
            sum_sasur_lu = a @ ((1 - sa[t] - su[t] - r[t]) * lambda_u[t])
            sum_sa_la = a @ (sa[t] * lambda_a[t])
            sum_su_mu = a @ (su[t] * mu[t])
            sum_sa_mu = a @ (sa[t] * mu[t])
            dlam_a, dlam_u, dmu = adj(lambda_a[t], lambda_u[t], mu[t], sum_sa, sum_su, sum_r, sum_sasur_la,
                                      sum_su_la, sum_su_lu, sum_sasur_lu, sum_sa_la, sum_su_mu, sum_sa_mu,beta1, beta2, eta, delta, gamma_a, gamma_u, omega)
            # =========================== backward update ============
            lambda_a[t - 1] = lambda_a[t] - h * dlam_a
            lambda_u[t - 1] = lambda_u[t] - h * dlam_u
            mu[t - 1] = mu[t] - h * dmu
        else:
            # =========================== nature evolution============
            # print("nonpulse forward t:", t)
            sum_sa = np.dot(a, sa[t])
            sum_su = np.dot(a, su[t])
            sum_r = np.dot(a, r[t])
            sum_sasur_la = a @ ((1 - sa[t] - su[t] - r[t]) * lambda_a[t])
            sum_su_la = a @ (su[t] * lambda_a[t])
            sum_su_lu = a @ (su[t] * lambda_u[t])
            sum_sasur_lu = a @ ((1 - sa[t] - su[t] - r[t]) * lambda_u[t])
            sum_sa_la = a @ (sa[t] * lambda_a[t])
            sum_su_mu = a @ (su[t] * mu[t])
            sum_sa_mu = a @ (sa[t] * mu[t])
            dlam_a, dlam_u, dmu = adj(lambda_a[t], lambda_u[t], mu[t], sum_sa, sum_su, sum_r, sum_sasur_la,
                                      sum_su_la, sum_su_lu, sum_sasur_lu, sum_sa_la, sum_su_mu, sum_sa_mu, beta1, beta2, eta, delta, gamma_a, gamma_u,omega)
            # =========================== backward update ============
            lambda_a[t - 1] = lambda_a[t] - h * dlam_a
            lambda_u[t - 1] = lambda_u[t] - h * dlam_u
            mu[t - 1] = mu[t] - h * dmu
    return lambda_a, lambda_u, mu

def loc_sim (a, b,t, lambda_a, lambda_u, mu, sa, su, cg, cf):
    values = []
    indexes = []
    for m in range(len(a)):
        val = [a[m] + b[n] + (lambda_a[t] * g_func(cg, a[m])) @ sa[t] + (lambda_u[t] * f_func(cf, b[n])) @ su[t] - ((mu[t] * g_func(cg, a[m])) @ sa[t] + (mu[t] * f_func(cf, b[n])) @ su[t]) for n in range(m, len(b))]
        minm1 = min(val)
        ind1 = val.index(minm1)
        values.append(minm1)
        indexes.append((m, m + ind1))
    minm2 = min(values)
    ind2 = values.index(minm2)
    return indexes[ind2]

# def policy_sim(tauk, lambda_a, lambda_u, mu, sa, su, ca, cu):
def policy_sim(lambda_a, lambda_u, mu, sa, su, ca, cu,t_interval, pulse_interval, a_low, a_upp, u_low, u_upp, cg, cf):
    for t in range(t_interval):
        # for tk in tauk:
        #     if tk == t: # accommulate sum (a single value not a vector)
            if t!=0 and t!=t_interval-1 and t % pulse_interval==0:
                # =================search optimal policy======================

                aa = np.arange(a_low, a_upp, 0.1)  ### paremeter
                bb = np.arange(u_low, u_upp, 0.1)
                locs = loc_sim(aa, bb, t, lambda_a, lambda_u, mu, sa, su, cg, cf)
                m = locs[0]
                n = locs[1]
                ca[t] = aa[m]
                cu[t] = bb[n]
    return ca, cu
def profit_sim (r, ca, cu, t_interval, pulse_interval, h, omega):
    sum_tmp = sum(r[t].sum(axis=0) for t in range(t_interval))
    sum_ca = sum(ca[t] for t in range(0, t_interval, pulse_interval)) ## 非脉冲时刻的值是0，故累加不影响
    sum_cu = sum(cu[t] for t in range(0, t_interval, pulse_interval))
    prof = omega * sum_tmp * h - sum_ca - sum_cu
    return prof


def check_non_negative(arr):
    for row in arr:
        for num in row:
            if num < 0:
                return False
    return True

if __name__ == '__main__':
    # Parameters
    T = 5# final time
    h = 0.01 # time step
    N = 849    # num of nodes
    cg=1
    cf=1

    beta1 = 0.001
    beta2 = 0.002
    eta = 0.001
    delta = 0.001
    gamma_a = 0.002
    gamma_u = 0.001
    omega = 0.01
    a_low = 0.1
    a_upp = 0.8
    u_low = 0.3
    u_upp = 1
    maxiter = 8

    t_interval = int(T / h) + 1  # 刻度

    # Vectors for temp value storage
    # sa = np.random.random((t_interval, N))  # return random floats in [0, 1)
    # su = np.random.random((t_interval, N))
    # r = np.random.random((t_interval, N))

    # Vectors for temp value storage
    sa = np.zeros((t_interval, N))  # initialize states; initialization can be same with ones in our algorithm
    su = np.zeros((t_interval, N))  # same
    r = np.zeros((t_interval, N))  # same
    ### update this code
    sa0 = 0.2
    su0 = 0.2
    r0 = 0.5

    for i in range(len(sa[0])):  # for loop for all nodes at t0
        sa[0][i] = sa0
    for i in range(len(su[0])):  # for loop for all nodes at t0
        su[0][i] = su0
    for i in range(len(r[0])):  # for loop for all nodes at t0
        r[0][i] = r0


    lambda_a = np.zeros((t_interval, N))
    lambda_u = np.zeros((t_interval, N))
    mu = np.zeros((t_interval, N))
    cu = np.zeros(t_interval)
    ca = np.zeros(t_interval)
    J = np.zeros(maxiter+1)

    # threshold = 1  ## (0.5, 1, 1.5,...)

    pulse_interval = 50

    fp = "./data/network_data/politician_edges.csv"
    a = adj_matrix(fp)
    # a = adj_matrix2(fp)


    # # 生成随机数组
    # a = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]  # return an integer number between start and end, including both end values

    ## 选择合适的脉冲时刻
    # tauk = pulse_time(threshold)[0]

    print("========================first iteration=========================")
    sa, su, r = forward(sa, su, r, ca, cu, t_interval, pulse_interval, a, beta1, beta2, eta, delta, gamma_a, gamma_u, h, g_func, f_func, cg, cf)

    # print(sa)
    ### 0 次迭代收益
    J[0] = profit_sim(r, ca, cu, t_interval, pulse_interval, h, omega)
    for iter in range(0, maxiter):
        print('================================backward====================')
        lambda_a, lambda_u, mu = backward(lambda_a, lambda_u, mu, ca, cu, sa, su, r, t_interval, pulse_interval, a,
                                              beta1, beta2, eta, delta, gamma_a, gamma_u, omega, h, cg, cf)
        print('=============================policy============================')
        ca, cu = policy_sim(lambda_a, lambda_u, mu, sa, su, ca, cu, t_interval, pulse_interval, a_low, a_upp, u_low,
                                u_upp, cg, cf)
        print('=============================forward====================')
        sa, su, r = forward(sa, su, r, ca, cu, t_interval, pulse_interval, a, beta1, beta2, eta, delta, gamma_a,
                                gamma_u, h, g_func, f_func, cg, cf)
        J[iter + 1] = profit_sim(r, ca, cu, t_interval, pulse_interval, h, omega)


    print(check_non_negative(sa))
    print(check_non_negative(su))
    print(check_non_negative(r))
    print(J)


# save various variables
    dfr = pd.DataFrame(r)
    dfsa = pd.DataFrame(sa)
    dfsu = pd.DataFrame(su)
    dfca = pd.DataFrame(ca)
    dfcu = pd.DataFrame(cu)
    dfla = pd.DataFrame(lambda_a)
    dflu = pd.DataFrame(lambda_u)
    dfmu = pd.DataFrame(mu)
    dfJ = pd.DataFrame(J)
    # dfr.to_csv(
    #     "./data/exp_data/toUse/ISaSuR_lessNodes/r_fixed_fb_" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
    #         gamma_u) + "_" + str(omega) + ".csv", index=False)
    # dfsa.to_csv("./data/exp_data/toUse/ISaSuR_lessNodes/sa_fixed_fb_" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
    #         gamma_u) + "_" + str(omega) + ".csv", index=False)
    # dfsu.to_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su_fixed_fb_" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
    #         gamma_u) + "_" + str(omega) + ".csv", index=False)
    # dfca.to_csv("./data/exp_data/toUse/ISaSuR_lessNodes/ca_fixed_fb_" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
    #         gamma_u) + "_" + str(omega) + ".csv", index=False)
    # dfcu.to_csv("./data/exp_data/toUse/ISaSuR_lessNodes/cu_fixed_fb_" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
    #         gamma_u) + "_" + str(omega) + ".csv", index=False)
    dfJ.to_csv("./data/exp_data/toUse/ISaSuR_lessNodes/J_fixed_fb_" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)

    x_axis = np.arange(maxiter + 1)
    y_axis = J
    plt.plot(x_axis,y_axis)
    # plt.legend(loc="upper right")
    plt.xlabel('Iteration')
    plt.ylabel('J')
    plt.show()







