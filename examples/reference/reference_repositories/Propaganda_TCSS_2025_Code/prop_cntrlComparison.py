import numpy as np
from math import*
import prop_propaganda as pro
import prop_network as pn

from prop_network import adj_matrix
import matplotlib.pyplot as plt
import random
import pandas as pd


def f_func(cf, x):
    return cf * 0.6 * x/(x+5)
def g_func(cg, x):
    return cg * (exp(0.5*x)-1)/(exp(x)+1)

def sys(sa_t, su_t, r_t, sum_sa, sum_su, sum_r, beta1, beta2, eta, delta, gamma_a, gamma_u):  # states of the node i at time t

    dsa_t = beta1 * (1 - sa_t - su_t - r_t) * sum_sa + eta * su_t * sum_sa - delta * sa_t - gamma_a * sa_t * sum_r

    dsu_t = beta2 * (1 - sa_t - su_t - r_t) * sum_su - eta * su_t * sum_sa - delta * su_t - gamma_u * su_t * sum_r

    dr_t = gamma_u * su_t * sum_r + gamma_a * sa_t * sum_r - delta * r_t

    return dsa_t, dsu_t, dr_t

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
            print("impulse sa=", sa)
            print("impulse su=", su)
        else:
            print("nonpulse forward t:", t)
            sum_sa = np.dot(a, sa[t])
            sum_su = np.dot(a, su[t])
            sum_r = np.dot(a, r[t])
            ### 向前更新三个状态变量
            dsa, dsu, dr = sys(sa[t], su[t], r[t], sum_sa, sum_su, sum_r, beta1, beta2, eta, delta, gamma_a, gamma_u)
            sa[t + 1] = sa[t] + h * dsa
            su[t + 1] = su[t] + h * dsu
            r[t + 1] = r[t] + h * dr
            print("nature sa=", sa)
            print("nature su=", su)
            # print("====nonpulse forward sates======= ")
            # print(sa[t])
            # print(su[t])
            # print(r[t])
            # print("*****************")
            # print(sa)
            # print(su)
            # print(r)
        return sa, su, r


if __name__ == '__main__':
    # Parameters
    T = 5  # final time
    h = 0.01  # time step
    N = 1000  # num of nodes
    cg = 1
    cf = 1

    beta1 = 0.001
    beta2 = 0.002
    eta = 0.001
    delta = 0.001
    gamma_a = 0.002
    gamma_u = 0.001
    omega = 0.01
    a_low = 0.1
    a_upp = 0.8
    # a_mid = 0.45
    u_low = 0.3
    u_upp = 1
    # u_mid = 0.65
    # m = 100
    m = 1

    t_interval = int(T / h) + 1  # 刻度

    # fp = "./data/network_data/soc-youtube.mtx"
    fp = "./data/network_data/rt-retweet-crawl.mtx"
    # fp = "./data/network_data/politician_edges.csv"
    # a = pn.adj_matrix(fp)  # csv
    a = pn.adj_matrix2(fp)  #mtx

    # ca = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/ca_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values.flatten()
    cu = pd.read_csv(
        "./data/exp_data/toUse/ISaSuR_lessNodes/cu_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values.flatten()
#
# ### exp 4: comparison between optimal impulsive strategy and random strategy
######   a = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    J_lst = []


    for _ in range(m):   # m is the iteration times, m=100 means it generates 100 random strategies
        print("m = ", _)

        # Vectors for temp value storage
        sa = np.zeros((t_interval, N))  # initialize states; initialization can be same with ones in our algorithm
        su = np.zeros((t_interval, N))  # same
        r = np.zeros((t_interval, N))   # same
      ### update this code
        sa0 = 0.2
        su0 = 0.2
        r0 = 0.5

        for i in range(len(sa[0])): # for loop for all nodes at t0
            sa[0][i] = sa0
        for i in range(len(su[0])): # for loop for all nodes at t0
            su[0][i] = su0
        for i in range(len(r[0])):  # for loop for all nodes at t0
            r[0][i] = r0

        lambda_a = np.zeros((t_interval, N))   # initialize co-states
        lambda_u = np.zeros((t_interval, N))
        mu = np.zeros((t_interval, N))

        # cu = np.zeros(t_interval)              # initialize control
        ca = np.zeros(t_interval)

        pulse_interval = 50

        # random strategy: for each impulse time, we randomly generate a control (impulse value)
        for t in range(t_interval-1):
            if t != 0 and t % pulse_interval == 0:
                ca[t] = np.random.uniform(low=a_low, high=a_upp)        # [low, high)
                # cu[t] = np.random.uniform(low=u_low, high=u_upp)

        sa, su, r = pro.forward(sa, su, r, ca, cu, t_interval, pulse_interval, a, beta1, beta2, eta, delta, gamma_a, gamma_u, h, g_func, f_func, cg, cf)
        J = pro.profit_sim(r, ca, cu, t_interval, pulse_interval, h, omega)
        J_lst.append(J)
        df_j = pd.DataFrame(J_lst)
        df_r = pd.DataFrame(r)
        df_sa = pd.DataFrame(sa)
        df_su = pd.DataFrame(su)
        df_j.to_csv("./data/exp_data/random/fixed_J_tw_caRan_mixed.csv", index=False)
        # df_r.to_csv("./data/exp_data/toUse/ISaSuR_heuristic/fixed_r_mid_yb.csv", index=False)
        # df_sa.to_csv("./data/exp_data/toUse/ISaSuR_heuristic/fixed_sa_mid_yb.csv", index=False)
        # df_su.to_csv("./data/exp_data/toUse/ISaSuR_heuristic/fixed_su_mid_yb.csv", index=False)
# =======================================loading our payoff, joining it to J_lst==========================================================
#     df1 = pd.read_csv("./data/exp_data/random/J4_yb.csv")
#     J_lst = df1['0'].tolist()
#     j = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/J4_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
#     J_lst.insert(0, j.values[-1])

# ===================================visualization of comparison between impulsive strategy and ransom strategy==========================
#     fig, axes = plt.subplots()
#     x = 1 + np.arange(m+1)
#     axes.stem(x, J_lst, linefmt='C3-', markerfmt='C3o')
#     axes.stem(x[1:], J_lst[1:], basefmt='C0-', markerfmt='C0o')
#     axes.set_xlabel('c', fontweight='bold')
#     axes.set_ylabel('J(c)', fontweight='bold')
#     y1 = [0, 20, 40, 60, 80, 100]
#     axes.set_xticks(y1, ['$c^{*}$', '$c_{20}$', '$c_{40}$', '$c_{60}$', '$c_{80}$', '$c_{100}$'])
#     plt.savefig("./data/exp3-4_fig/profit_comparison_yb_4.pdf")

    # plt.show()

    #
    # # ## #### exp 3:   control strategy
    # ca_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/ca4_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
    # cu_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/cu4_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
    # ca_rt = pd.read_csv(
    #     "./data/exp_data/toUse/ISaSuR_lessNodes/ca4_rt_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
    # cu_rt = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/cu4_rt_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
    # ca_yb = pd.read_csv(
    #     "./data/exp_data/toUse/ISaSuR_lessNodes/ca4_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
    # cu_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/cu4_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
    # #
    # #
    # #
    # fig, axes = plt.subplots(3, 2, figsize=(9, 5.5), sharey=True, sharex=True)
    # x_axis = np.arange(0, T + 0.01, 0.01)
    #
    # axes[0, 0].plot(x_axis, ca_fb, 'tab:pink')
    # axes[0, 0].set_title("On Facebook", fontsize='8')
    # axes[0, 0].set_ylim([-0.02, 0.95])
    # axes[0, 0].set_ylabel('$c^a(t)$')
    # #
    # axes[0, 1].plot(x_axis, cu_fb, 'tab:green')
    # axes[0, 1].set_title("On Facebook", fontsize='8')
    # axes[0, 1].set_ylabel('$c^u(t)$')
    #
    # axes[1, 0].plot(x_axis, ca_rt, 'tab:pink')
    # axes[1, 0].set_title("On Twitter", fontsize='8')
    # axes[1, 0].set_ylabel('$c^a(t)$')
    #
    # axes[1, 1].plot(x_axis, cu_rt, 'tab:green')
    # axes[1, 1].set_title("On Twitter", fontsize='8')
    # axes[1, 1].set_ylabel('$c^u(t)$')
    #
    # axes[2, 0].plot(x_axis, ca_yb, 'tab:pink')
    # axes[2, 0].set_title("On YouTube", fontsize='8')
    # axes[2, 0].set_ylabel('$c^a(t)$')
    # axes[2, 0].set_xlabel('t')
    #
    # axes[2, 1].plot(x_axis, cu_yb, 'tab:green')
    # axes[2, 1].set_title("On YouTube", fontsize='8')
    # axes[2, 1].set_ylabel('$c^u(t)$')
    # axes[2, 1].set_xlabel('t')
    # #
    # plt.savefig("./data/exp3-4_fig/control_stategy_4.pdf" )
    # # plt.show()
    #
