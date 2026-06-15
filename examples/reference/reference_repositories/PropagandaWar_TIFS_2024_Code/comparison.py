import numpy as np
from igraph import *
import matplotlib.pyplot as plt
from math import *
import random
import pandas as pd
from demo_network import my_degree
from demo_network import graphs

#
# def fr (x):
#     return c_b * x/(x+1)
# def gr (x):
#     return c_b * x/(x+1)
# def fb (x):
#     return c_r * log(x+1)
# def gb (x):
#     return c_r * log(x+1)


def fr (x):
    return c_b * x/(x+1)
def gr (x):
    return c_b * x/(x+1)
def fb (x):
    return c_r * sqrt(x)
def gb (x):
    return c_r * sqrt(x)

def theta(k,pk,state_t):
    x= np.dot(state_t * pk, k)/(np.dot(k,pk))
    return x


def sysRed(ub_t, pr_t, cr_t, theta_r1, theta_r2):  # states of the node with k degree at time t
    dpr_t = (fr(ub_t) + kr * beta_r * theta_r1) * (1 - pr_t - cr_t) - (delta_r + kr * gamma_r2 * theta_r2) * pr_t
    dcr_t = kr * gamma_r1 * theta_r2 * (1 - pr_t - cr_t) + kr * gamma_r2 * theta_r2 * pr_t - delta_r * cr_t
    return dpr_t, dcr_t

def sysBlue(ur_t, pb_t, cb_t, theta_b1, theta_b2):
    dpb_t = (fb(ur_t) + kb * beta_b * theta_b1) * (1 - pb_t - cb_t) - (delta_b + kb * gamma_b2 * theta_b2) * pb_t
    dcb_t = kb * gamma_b1 * theta_b2 * (1 - pb_t - cb_t) + kb * gamma_b2 * theta_b2 * pb_t - delta_b * cb_t
    return dpb_t, dcb_t

def forwardRed(pr, cr, ub, vr):
    for t in range(t_interval-1): #-1 due to forward
        if t != 0 and t % pulse_interval_r == 0:   ###
            #============== impulse control for cp at pulse times ===============
            pr[t] = pr[t] - gr(vr[t]) * pr[t]
            cr[t] = cr[t] + gr(vr[t]) * (1 - cr[t])
            #=============== continuous control for p at times other than pulse times ================
            theta_r1 = theta(kr, pkr, pr[t])
            theta_r2 = theta(kr, pkr, cr[t])
            dpr, dcr = sysRed(ub[t], pr[t], cr[t], theta_r1, theta_r2)
            pr[t+1] = pr[t] + h * dpr
            cr[t+1] = cr[t] + h * dcr
        else:
            # =============== continuous control for p at times other than pulse times ================
            theta_r1 = theta(kr, pkr, pr[t])
            theta_r2 = theta(kr, pkr, cr[t])
            dpr, dcr = sysRed(ub[t], pr[t], cr[t], theta_r1, theta_r2)
            pr[t + 1] = pr[t] + h * dpr
            cr[t + 1] = cr[t] + h * dcr
    return pr, cr

def forwardBlue(pb, cb, ur, vb):
    for t in range(t_interval-1): #-1 due to forward
        if t != 0 and t % pulse_interval_b == 0:   ###
            #============== impulse control for cp at pulse times ===============
            pb[t] = pb[t] - gb(vb[t]) * pb[t]
            cb[t] = cb[t] + gb(vb[t]) * (1 - cb[t])
            #=============== continuous control for p at times other than pulse times ================
            theta_b1 = theta(kb, pkb, pb[t])
            theta_b2 = theta(kb, pkb, cb[t])
            dpb, dcb = sysBlue(ur[t], pb[t], cb[t], theta_b1, theta_b2)
            pb[t+1] = pb[t] + h * dpb
            cb[t+1] = cb[t] + h * dcb
        else:
            # =============== continuous control for p at times other than pulse times ================
            theta_b1 = theta(kb, pkb, pb[t])
            theta_b2 = theta(kb, pkb, cb[t])
            dpb, dcb = sysBlue(ur[t], pb[t], cb[t], theta_b1, theta_b2)
            pb[t + 1] = pb[t] + h * dpb
            cb[t + 1] = cb[t] + h * dcb
    return pb, cb

def randomRed(ur, vr):
    for t in range(t_interval - 1):  # -1 due to forward
        if t != 0 and t % pulse_interval_r == 0:  ## impulse times
            # ============== impulse control for cp at pulse times ===============
            vr[t] = np.random.uniform(low=vr_low, high=vr_upp)
            ur[t] = np.random.uniform(low=ur_low, high=ur_upp)
        else:
            # =============== continuous control for p at times other than pulse times ================
            ur[t] = np.random.uniform(low=ur_low, high=ur_upp)
    return vr, ur

def randomBlue(ub, vb):
    for t in range(t_interval - 1):  # -1 due to forward
        if t != 0 and t % pulse_interval_b == 0:  ## impulse times
            # ============== impulse control for cp at pulse times ===============
            vb[t] = np.random.uniform(low=vb_low, high=vb_upp)
            ub[t] = np.random.uniform(low=ub_low, high=ub_upp)
        else:
            # =============== continuous control for p at times other than pulse times ================
            ub[t] = np.random.uniform(low=ub_low, high=ub_upp)
    return vb, ub



def payoffRed(pb, pr, cr, ur, vr):
    sum_pb = sum(pb[t].sum(axis=0) for t in range(t_interval))
    sum_cr = sum(cr[t].sum(axis=0) for t in range(t_interval))
    sum_pr = sum(pr[t].sum(axis=0) for t in range(t_interval))
    sum_ur = sum(ur)    # condition is the initialization is zero
    sum_vr = sum(vr)    # condition is the initialization is zero
    jred = (brp * sum_pb + brc * sum_cr - lr * sum_pr - sum_ur) * h - sum_vr   # note this value maybe negative
    return jred

def payoffBlue(pr, pb, cb, ub, vb):
    sum_pr = sum(pr[t].sum(axis=0) for t in range(t_interval))
    sum_cb = sum(cb[t].sum(axis=0) for t in range(t_interval))
    sum_pb = sum(pb[t].sum(axis=0) for t in range(t_interval))
    sum_ub = sum(ub)    # condition is the initialization is zero
    sum_vb = sum(vb)    # condition is the initialization is zero
    jblue = (bbp * sum_pr + bbc * sum_cb - lb * sum_pb - sum_ub) * h - sum_vb   # note this value maybe negative
    return jblue

def funRed():
    for _ in range(m):
        print("m=", _)
        ## initialized states
        pr = np.zeros((t_interval, len(kr)))
        cr = np.zeros((t_interval, len(kr)))
        pb = np.zeros((t_interval, len(kb)))
        cb = np.zeros((t_interval, len(kb)))
        pr0 = 0.3
        cr0 = 0.4
        pb0 = 0.1
        cb0 = 0.2
        for i in range(len(pr[0])):
            pr[0][i] = pr0
        for i in range(len(cr[0])):
            cr[0][i] = cr0
        for i in range(len(pb[0])):
            pb[0][i] = pb0
        for i in range(len(cb[0])):
            cb[0][i] = cb0

        ## initialized strategy
        ur = np.zeros(t_interval)
        vr = np.zeros(t_interval)
        # ub = np.zeros(t_interval)
        # vb = np.zeros(t_interval)
        ## payoffBlue: ur and vr for NE but ub and vb for random strategy
        pub = pd.read_csv("./result/result_new33/ub_" + a + str(b)+".csv")
        pvb = pd.read_csv("./result/result_new33/vb_" + a + str(b)+".csv")
        ub = pub.values.flatten()
        vb = pvb.values.flatten()
        vr, ur = randomRed(ur, vr)
        ## state updates
        pr, cr = forwardRed(pr, cr, ub, vr)
        pb, cb = forwardBlue(pb, cb, ur, vb)
        ## payoff
        jr = payoffRed(pb, pr, cr, ur, vr)
        jrl[_] = jr

# ###### add  sates
#         apr = np.zeros(t_interval)
#         acr = np.zeros(t_interval)
#         for t in range(t_interval):
#             apr[t] = np.dot(pkr, pr[t])  ### generate 1 weighted average for each _
#         for t in range(t_interval):
#             acr[t] = np.dot(pkr, cr[t])
#         print("apr=", apr)
#         print("acr=", acr)
#         prl[_] = apr[-1]
#         crl[_] = acr[-1]
#         print("minimum of apr=", prl[_])
#         print("maximum of acr=", crl[_])

    return jrl

def funBlue():
    for _ in range(m):
        print("m=", _)
        ## initialized states
        pr = np.zeros((t_interval, len(kr)))
        cr = np.zeros((t_interval, len(kr)))
        pb = np.zeros((t_interval, len(kb)))
        cb = np.zeros((t_interval, len(kb)))
        pr0 = 0.3
        cr0 = 0.4
        pb0 = 0.1
        cb0 = 0.2
        for i in range(len(pr[0])):
            pr[0][i] = pr0
        for i in range(len(cr[0])):
            cr[0][i] = cr0
        for i in range(len(pb[0])):
            pb[0][i] = pb0
        for i in range(len(cb[0])):
            cb[0][i] = cb0

        ## initialized strategy
        # ur = np.zeros(t_interval)
        # vr = np.zeros(t_interval)
        ub = np.zeros(t_interval)
        vb = np.zeros(t_interval)
        ## payoffBlue: ur and vr for NE but ub and vb for random strategy
        pur = pd.read_csv("./result/result_new33/ur_" + a + str(b)+".csv")
        pvr = pd.read_csv("./result/result_new33/vr_" + a + str(b)+".csv")
        ur = pur.values.flatten()
        vr = pvr.values.flatten()
        vb, ub = randomBlue(ub, vb)
        ## state updates
        pr, cr = forwardRed(pr, cr, ub, vr)
        pb, cb = forwardBlue(pb, cb, ur, vb)
        # payoff
        jb = payoffBlue(pr, pb, cb, ub, vb)
        jbl[_] = jb


        # ###### add  sates late
        # for t in range(t_interval):
        #     pbl[_] = np.min(np.dot(pkb, pb[t]))  ### generate 1 weighted average for each _
        # for t in range(t_interval):
        #     cbl[_] = np.max(np.dot(pkb, cb[t]))

    return jbl


if __name__ == '__main__':
    ######################### need the network function to get k and pk #############################?????????????
    # Create graph

    # fpr = "./data/socfb-Bingham82.mtx"
    # fpb = "./data/socfb-Cal65.mtx"
    #
    fpr = "./data/rt_barackobama.edges"  # 10k
    fpb = "./data/rt-pol.txt"  # 18k
    Gr = graphs(fpr)
    Gb = graphs(fpb)

    kr, pkr = my_degree(Gr)
    kb, pkb = my_degree(Gb)
    #
    T = 5
    h = 0.001
    t_interval = int(T / h) + 1  # 总刻度， 故下标应该是[0, t_interval-1]
    # pulse_interval_b = 400  # fb setting
    # pulse_interval_r = 500
    pulse_interval_b = 300    # tw setting
    pulse_interval_r = 400

## tw parameters (matches the twitter datasets selected above)
    beta_r = 0.03
    gamma_r1 = 0.04
    gamma_r2 = 0.02
    beta_b = 0.04
    gamma_b1 = 0.05
    gamma_b2 = 0.03

    delta_r = 0.02
    delta_b = 0.01

    c_b = 0.1
    c_r = 0.06

    lowr = 1
    uppr = 8
    lowb = 3
    uppb = 10
    coeff = 1
    brp = brc = lr = bbp = bbc = lb = coeff

    ur_low = lowr
    ur_upp = uppr
    vr_low = lowr
    vr_upp = uppr
    ub_low = lowb
    ub_upp = uppb
    vb_low = lowb
    vb_upp = uppb

    a = 'tw_coeff_'
    b = 1

    m = 100
    # ## initial all variables (states, co_states, controls, payoffs)
    #
    jrl = np.zeros(m)  #[0, m-1]
    jbl = np.zeros(m)  # [0, m-1]


    team = input("which team you want to compare?")
    if team == 'r':

        Jr = funRed()
        df_r = pd.DataFrame(Jr)
        df_r.to_csv("./result/random/random_jr_" + a + str(b)+".csv", index=False)
    elif team == 'b':
        Jb = funBlue()
        df_b = pd.DataFrame(Jb)
        df_b.to_csv("./result/random/random_jb_" + a + str(b)+".csv", index=False)







    # # # # #### caculate weighted average of states
    # # #
    # pr = pd.read_csv("./result/result_new33/pr_" + a + str(b)+".csv").values
    # cr = pd.read_csv("./result/result_new33/cr_" + a + str(b)+".csv").values
    # pb = pd.read_csv("./result/result_new33/pb_" + a + str(b)+".csv").values
    # cb = pd.read_csv("./result/result_new33/cb_" + a + str(b)+".csv").values
    #
    # ne_pr = np.zeros(t_interval)
    # ne_cr = np.zeros(t_interval)
    # ne_pb = np.zeros(t_interval)
    # ne_cb = np.zeros(t_interval)
    # for t in range(t_interval):
    #     ne_pr[t] = np.dot(pkr, pr[t])
    # for t in range(t_interval):
    #     ne_cr[t] = np.dot(pkr, cr[t])
    # for t in range(t_interval):
    #     ne_pb[t] = np.dot(pkb, pb[t])
    # for t in range(t_interval):
    #     ne_cb[t] = np.dot(pkb, cb[t])
    #
    # print("min_ne_pr=",ne_pr[-1])
    # print("max_ne_cr=", ne_cr[-1])
    #
    # x2 = np.arange(t_interval) * 0.001
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    #
    # ax[0].plot(x2, ne_cr, 'r', label='avg_Cr')
    # ax[0].plot(x2, ne_pr, 'r--', label='avg_Pr')
    # ax[0].legend(loc="upper left")
    # ax[0].set_ylabel("Average States for Red Team")
    # ax[0].set_xlabel("Time")
    #
    # ax[1].plot(x2, ne_cb, 'b', label='avg_Cb')
    # ax[1].plot(x2, ne_pb, 'b--', label='avg_Pb')
    # ax[1].legend(loc="upper left")
    # ax[1].set_ylabel("Average States for Blue Team")
    # ax[1].set_xlabel("Time")
    # # plt.show()
    # plt.savefig("./result/figs/avgStates/states22_" + a + str(b)+".pdf")



  ## controls

    # ur = pd.read_csv("./result/result_new33/ur_" + a + str(b)+".csv").values
    # vr = pd.read_csv("./result/result_new33/vr_" + a + str(b)+".csv").values
    # ub = pd.read_csv("./result/result_new33/ub_" + a + str(b)+".csv").values
    # vb = pd.read_csv("./result/result_new33/vb_" + a + str(b)+".csv").values
    #
    # x2 = np.arange(t_interval) * 0.001
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
    #
    # ax[0][0].plot(x2, ur, 'r')
    # ax[0][0].set_ylim([0,np.max(ur) + 0.5])
    # ax[0][0].set_ylabel("ur")
    #
    #
    # ax[0][1].plot(x2, ub, 'b')
    # ax[0][1].set_ylim([2, np.max(ub) + 0.5])
    # ax[0][1].set_ylabel("ub")
    #
    #
    # ax[1][0].bar(x2, vr.flatten(), width=0.15,  linewidth=1, color='r',alpha=0.6)
    # ax[1][0].set_ylabel("vr")
    # ax[1][0].set_xlabel("Time")
    #
    # ax[1][1].bar(x2, vb.flatten(), width=0.15, linewidth=1, color='b', alpha=0.6)
    # ax[1][1].set_ylabel("vb")
    # ax[1][1].set_xlabel("Time")
    # # #
    # plt.savefig("./result/figs/controls/controls_22" + a + str(b)+".pdf")
    # #

## control and states — requires precomputed NE results from propWar.py runs.
## Disabled by default; set `inspect_ne = True` after running propWar.py to populate ./result/result_new33/.
    inspect_ne = False
    if inspect_ne:
        pr = pd.read_csv("./result/result_new33/pr_" + a + str(b) + ".csv").values
        cr = pd.read_csv("./result/result_new33/cr_" + a + str(b)+".csv").values
        pb = pd.read_csv("./result/result_new33/pb_" + a + str(b)+".csv").values
        cb = pd.read_csv("./result/result_new33/cb_" + a + str(b)+".csv").values

        ur = pd.read_csv("./result/result_new33/ur_" + a + str(b) + ".csv").values
        vr = pd.read_csv("./result/result_new33/vr_" + a + str(b) + ".csv").values
        ub = pd.read_csv("./result/result_new33/ub_" + a + str(b) + ".csv").values
        vb = pd.read_csv("./result/result_new33/vb_" + a + str(b) + ".csv").values

        ne_pr = np.zeros(t_interval)
        ne_cr = np.zeros(t_interval)
        ne_pb = np.zeros(t_interval)
        ne_cb = np.zeros(t_interval)
        for t in range(t_interval):
            ne_pr[t] = np.dot(pkr, pr[t])
        for t in range(t_interval):
            ne_cr[t] = np.dot(pkr, cr[t])
        for t in range(t_interval):
            ne_pb[t] = np.dot(pkb, pb[t])
        for t in range(t_interval):
            ne_cb[t] = np.dot(pkb, cb[t])

        print("pr=", ne_pr)
        print("cr=", ne_cr)
        print("pb=", ne_pb)
        print("cb=", ne_cb)

    # def addlabels(x, y):
    #     for i in range(len(x)):
    #         plt.text(i, y[i], y[i], ha='center', size=3)

    # x2 = np.arange(t_interval) * 0.001
    # fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,10))
    #
    # ax[0][0].plot(x2, ur, 'r')
    # ax[0][0].set_ylim([0, np.max(ur) + 0.5])
    # ax[0][0].set_ylabel(r'$u_r(t)$')
    #
    # ax[0][1].plot(x2, ub, 'b')
    # ax[0][1].set_ylim([2, np.max(ub) + 0.5])
    # ax[0][1].set_ylabel(r'$u_b(t)$')
    #
    # ax[1][0].bar(x2, vr.flatten(), width=0.1, linewidth=1, color='r', alpha=0.6)
    # for i, value in enumerate(vr.flatten()):
    #     if value != 0:
    #         ax[1][0].text(x2[i], value, str(np.round(value,1)), ha='center', va='bottom')
    # ax[1][0].set_ylabel(r'$v_r(\tau_i)$')
    #
    #
    # ax[1][1].bar(x2, vb.flatten(), width=0.1, linewidth=1, color='b', alpha=0.6)
    # for i, value in enumerate(vb.flatten()):
    #     if value != 0:
    #         ax[1][1].text(x2[i], value, str(np.round(value, 1)), ha='center', va='bottom')
    # ax[1][1].set_ylabel(r'$v_b(\tau_i)$')
    #
    #
    # ax[2][0].plot(x2, ne_cr, 'r', label='avg_Cr')
    # ax[2][0].plot(x2, ne_pr, 'r--', label='avg_Pr')
    # ax[2][0].legend(loc="lower right")
    # ax[2][0].set_ylabel(r'$C_r(t),P_r(t)$')
    # ax[2][0].set_xlabel("Time")
    #
    # ax[2][1].plot(x2, ne_cb, 'b', label='avg_Cb')
    # ax[2][1].plot(x2, ne_pb, 'b--', label='avg_Pb')
    # ax[2][1].legend(loc="lower right")
    # ax[2][1].set_ylabel(r'$C_b(t),P_b(t)$')
    # ax[2][1].set_xlabel("Time")
    #
    # plt.savefig("./result/figs/avgStates/states_contr_" + a + str(b)+".pdf")
    #
    #
    # plt.show()