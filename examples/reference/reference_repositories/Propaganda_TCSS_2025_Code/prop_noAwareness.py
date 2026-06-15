import numpy as np
import math
from math import *
from prop_network import adj_matrix
from prop_network import adj_matrix2
import matplotlib.pyplot as plt
import random
import pandas as pd
import prop_propaganda as pro

def g_func(cg, x):
    return cg * 0.06 * sqrt(x)

def f_func(cf, x):
    return cf * 0.5 /pi * np.arctan(0.5*x)


def loc_sim(b, t, lambda_a, lambda_u, mu, sa, su, cg, cf):

    # for m in range(len(a)):
    val = [b[n] + (lambda_a[t] * g_func(cg, 0)) @ sa[t] + (lambda_u[t] * f_func(cf, b[n])) @ su[t] - ((mu[t] * g_func(cg, 0)) @ sa[t] + (mu[t] * f_func(cf, b[n])) @ su[t]) for n in range(len(b))]
    minm1 = min(val)
    ind1 = val.index(minm1)
    # values.append(minm1)
    # indexes.append(ind1)
    # minm2 = min(values)
    # ind2 = values.index(minm2)
    return ind1


def policy_sim(lambda_a, lambda_u, mu, sa, su, cu, ca, t_interval, pulse_interval, a_low, a_upp, u_low, u_upp, cg, cf):
    for t in range(t_interval):
        # for tk in tauk:
        #     if tk == t: # accommulate sum (a single value not a vector)
            if t!=0 and t!=t_interval-1 and t % pulse_interval==0:
                # =================search optimal policy======================
                # aa = np.arange(a_low, a_upp, 0)  ### paremeter
                bb = np.arange(u_low, u_upp, 0.1)
                locs = loc_sim(bb, t, lambda_a, lambda_u, mu, sa, su, cg, cf)
                # m = locs[0]
                n = locs
                # ca[t] = aa[m]
                cu[t] = bb[n]
                # print("ca=", ca[t])
                # print("cu=", cu[t])
    return ca, cu




if __name__ == '__main__':
    # Parameters
    T = 5 # final time
    h = 0.01 # time step
    # K = 3  # num of impulse
    N = 849 # num of nodes
    cg = 0
    cf = 1

    beta1 = 0  # the branch of sa is 0
    beta2 = 0.002
    eta = 0
    delta = 0.001
    gamma_a = 0
    gamma_u = 0.001
    omega = 0.01
    a_low = 0
    a_upp = 0
    u_low = 0.4
    u_upp = 1.8
    maxiter = 10
    t_interval = int(T / h) + 1  # 刻度

    # Vectors for temp value storage
    sa = np.zeros((t_interval, N))  # return random floats in [0, 1)
    su = np.random.random((t_interval, N))
    r = np.random.random((t_interval, N))

    lambda_a = np.zeros((t_interval, N))
    lambda_u = np.zeros((t_interval, N))
    mu = np.zeros((t_interval, N))
    cu = np.zeros(t_interval)
    ca = np.zeros(t_interval)
    J = np.zeros(maxiter+1)

    # threshold = 1  ## (0.5, 1, 1.5,...)
    pulse_interval = 50

    fp = "./data/network_data/politician_edges.csv"
    a = adj_matrix(fp) # csv
    # a = adj_matrix2(fp)  #mtx

    # 生成随机数组
    # a = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]  # return an integer number between start and end, including both end values


    print("========================first iteration=========================")
    sa, su, r = pro.forward(sa, su, r, ca, cu, t_interval, pulse_interval, a, beta1, beta2, eta, delta, gamma_a, gamma_u,h, g_func, f_func, cg, cf)

    # print(sa)
    ### 0 次迭代收益
    J[0] = pro.profit_sim(r, ca, cu, t_interval, pulse_interval, h, omega)
    for iter in range(0, maxiter):
        print('================================backward====================')
        lambda_a, lambda_u, mu = pro.backward(lambda_a, lambda_u, mu, ca, cu, sa, su, r, t_interval, pulse_interval, a, beta1, beta2, eta, delta, gamma_a, gamma_u, omega, h, cg,cf)
        print('=============================policy============================')
        ca, cu = policy_sim(lambda_a, lambda_u, mu, sa, su, cu, ca, t_interval, pulse_interval, a_low, a_upp, u_low, u_upp, cg, cf)
        print('=============================forward====================')
        sa, su, r = pro.forward(sa, su, r, ca, cu, t_interval, pulse_interval, a, beta1, beta2, eta, delta, gamma_a, gamma_u, h, g_func, f_func, cg, cf)
        J[iter+1] = pro.profit_sim(r, ca, cu, t_interval, pulse_interval, h, omega)

# save various variables
    dfr = pd.DataFrame(r)
    dfsa = pd.DataFrame(sa)
    dfsu = pd.DataFrame(su)
    dfca = pd.DataFrame(ca)
    dfcu = pd.DataFrame(cu)
    dfla = pd.DataFrame(lambda_a)
    dflu = pd.DataFrame(lambda_u)
    dfJ = pd.DataFrame(J)
    dfr.to_csv("./data/exp_data/toUse/SIR_lessNodes/r4_fb" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)
    dfsa.to_csv("./data/exp_data/toUse/SIR_lessNodes/sa4_fb" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)
    dfsu.to_csv("./data/exp_data/toUse/SIR_lessNodes/su4_fb" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)
    dfca.to_csv("./data/exp_data/toUse/SIR_lessNodes/ca4_fb" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)
    dfcu.to_csv("./data/exp_data/toUse/SIR_lessNodes/cu4_fb" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)
    dfla.to_csv("./data/exp_data/toUse/SIR_lessNodes/la4_fb" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)
    dflu.to_csv("./data/exp_data/toUse/SIR_lessNodes/lu4_fb" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)
    dfJ.to_csv("./data/exp_data/toUse/SIR_lessNodes/J4_fb" + str(beta1) + str(beta2) + "_" + str(eta) + "_" + str(delta) + "_" + str(gamma_a) + str(
            gamma_u) + "_" + str(omega) + ".csv", index=False)

    x_axis = np.arange(maxiter + 1)
    y_axis = J
    plt.plot(x_axis,y_axis)
    # plt.legend(loc="upper right")
    plt.xlabel('Iteration')
    plt.ylabel('J')
    plt.show()






