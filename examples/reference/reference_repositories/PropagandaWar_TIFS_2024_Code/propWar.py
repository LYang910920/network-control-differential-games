import numpy as np
from igraph import *
from math import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import time

from demo_network import my_degree

from demo_network import graphs

from demo_network import graphs_tmx

# theta_r1 = np.zeros(t_interval)
# theta_r2 = np.zeros(t_interval)

# twitter:
# def fr (x):
#     return c_r * x/(x+1)
# def gr (x):
#     return c_r * x/(x+1)
# def fb (x):
#     return c_b * sqrt(x)
# def gb (x):
#     return c_b * sqrt(x)

# facebook
def fr (x):
    return c_r * x/(x+1)
def gr (x):
    return c_r * x/(x+1)
def fb (x):
    return c_b * log(x+1)
def gb (x):
    return c_b * log(x+1)





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


def forwardRed():
    for t in range(t_interval-1): #-1 due to forward
        if t != 0 and t % pulse_interval_r == 0:   ###
        # if t in pulse_red_lst:
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

def forwardBlue():
    for t in range(t_interval-1): #-1 due to forward
        if t != 0 and t % pulse_interval_b == 0:
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


def adjRed(pr_t, pb_t, cr_t, cb_t, ur_t, ub_t, lambda_rt, lambda_bt, mu_rt, mu_bt, theta_r1, theta_r2, theta_b1, theta_b2):
    dlam_rt = lr + (kr * beta_r * (theta_r1 - kr * pkr/avd_r * (1 - pr_t - cr_t)) + fr(ub_t) + delta_r + kr * gamma_r2 * theta_r2) * lambda_rt + kr * theta_r2 * (gamma_r1-gamma_r2) * mu_rt
    dmu_rt = -brc + (fr(ub_t) + kr * beta_r * theta_r1) * lambda_rt + gamma_r2 * pr_t * kr**2 * pkr/avd_r * (lambda_rt - mu_rt) + gamma_r1 * kr * (theta_r2 - kr * pkr/avd_r * (1 - pr_t - cr_t)) * mu_rt + delta_r * mu_rt
    dlam_bt = -brp + (kb * beta_b * (theta_b1 - kb * pkb/avd_b * (1 - pb_t - cb_t)) + fb(ur_t) + delta_b + kb * gamma_b2 * theta_b2) * lambda_bt + kb * theta_b2 * (gamma_b1 - gamma_b2) * mu_bt
    dmu_bt = (fb(ur_t) + kb * beta_b * theta_b1) * lambda_bt + gamma_b2 * pb_t * kb**2 * pkb/avd_b * (lambda_bt - mu_bt) + (delta_b + kb * gamma_b1 * (theta_b2 - kb * pkb/avd_b * (1 - pb_t - cb_t))) * mu_bt
    return dlam_rt, dmu_rt, dlam_bt, dmu_bt

def adjBlue(pr_t, pb_t, cr_t, cb_t, ur_t, ub_t, phi_rt, phi_bt, psi_rt, psi_bt, theta_r1, theta_r2, theta_b1, theta_b2):
    dphi_rt = -bbp + (kr * beta_r * (theta_r1 - kr * pkr/avd_r * (1 - pr_t - cr_t)) + fr(ub_t) + delta_r + kr * gamma_r2 * theta_r2) * phi_rt + kr * theta_r2 * (gamma_r1-gamma_r2) * psi_rt
    dpsi_rt = (fr(ub_t) + kr * beta_r * theta_r1) * phi_rt + gamma_r2 * pr_t * (kr**2 * pkr/avd_r) * (phi_rt - psi_rt) + (delta_r + kr * gamma_r1 * (theta_r2 - kr * pkr/avd_r * (1 - pr_t - cr_t))) * psi_rt
    dphi_bt = lb + (kb * beta_b * (theta_b1 - kb * pkb/avd_b * (1 - pb_t - cb_t)) + fb(ur_t) + delta_b + kb * gamma_b2 * theta_b2) * phi_bt + kb * theta_b2 * (gamma_b1-gamma_b2) * psi_bt
    dpsi_bt = -bbc + (fb(ur_t) + kb * beta_b * theta_b1) * phi_bt + gamma_b2 * pb_t * (kb**2 * pkb/avd_b) * (phi_bt - psi_bt) + gamma_b1 * kb * (theta_b2 - kb * pkb/avd_b * (1 - pb_t - cb_t)) * psi_bt + delta_b * psi_bt

    return dphi_rt, dpsi_rt, dphi_bt, dpsi_bt

def backwardRed():
    for t in range(t_interval-1, 0, -1):  # t is in [t_interval-1, 1]
        theta_r1 = theta(kr, pkr, pr[t])
        theta_r2 = theta(kr, pkr, cr[t])
        theta_b1 = theta(kb, pkb, pb[t])
        theta_b2 = theta(kb, pkb, cb[t])
        if t != t_interval-1 and t % pulse_interval_r == 0:
        # if t in pulse_red_lst:
            # =========================== adjoint evolution at pulse times============
            lambda_r[t] = lambda_r[t] - gr(vr[t]) * lambda_r[t]
            mu_r[t] = mu_r[t] - gr(vr[t]) * mu_r[t]
            lambda_b[t] = lambda_b[t] - gb(vb[t]) * lambda_b[t]
            mu_b[t] = mu_b[t] - gb(vb[t]) * mu_b[t]
            # print("vr==", vr[t])
            # print("gr==", gr(vr[t]))
            # =========================== adjoint evolution at times other than pulse times ============

            dlam_r, dmu_r, dlam_b, dmu_b = adjRed(pr[t], pb[t], cr[t], cb[t], ur[t], ub[t], lambda_r[t], lambda_b[t], mu_r[t], mu_b[t], theta_r1,  theta_r2, theta_b1, theta_b2)
            lambda_r[t-1] = lambda_r[t] - h * dlam_r
            mu_r[t-1] = mu_r[t] - h * dmu_r
            lambda_b[t-1] = lambda_b[t] - h * dlam_b
            mu_b[t-1] = mu_b[t] - h * dmu_b
            # print("gr(vr)==", gr(vr[t]))
            # print("gb(vb)==", gb(vb[t]))
        else:
            # =========================== adjoint evolution at times other than pulse times ============
            dlam_r, dmu_r, dlam_b, dmu_b = adjRed(pr[t], pb[t], cr[t], cb[t], ur[t], ub[t], lambda_r[t], lambda_b[t],mu_r[t], mu_b[t], theta_r1, theta_r2, theta_b1, theta_b2)
            lambda_r[t - 1] = lambda_r[t] - h * dlam_r
            mu_r[t - 1] = mu_r[t] - h * dmu_r
            lambda_b[t - 1] = lambda_b[t] - h * dlam_b
            mu_b[t - 1] = mu_b[t] - h * dmu_b
    return lambda_r, mu_r, lambda_b, mu_b

def backwardBlue():
    for t in range(t_interval-1, 0, -1):  # t is in [t_interval-1, 1]
        theta_r1 = theta(kr, pkr, pr[t])
        theta_r2 = theta(kr, pkr, cr[t])
        theta_b1 = theta(kb, pkb, pb[t])
        theta_b2 = theta(kb, pkb, cb[t])
        if t != t_interval-1 and t % pulse_interval_b == 0:
            # =========================== adjoint evolution at pulse times============
            phi_r[t] = phi_r[t] - gr(vr[t]) * phi_r[t]
            psi_r[t] = psi_r[t] - gr(vr[t]) * psi_r[t]
            phi_b[t] = phi_b[t] - gb(vb[t]) * phi_b[t]
            psi_b[t] = psi_b[t] - gb(vb[t]) * psi_b[t]
            # print("vb==", vb[t])
            # print("gb==", gb(vb[t]))
            # =========================== adjoint evolution at times other thab pulse times ============

            dphi_r, dpsi_r, dphi_b, dpsi_b = adjBlue(pr[t], pb[t], cr[t], cb[t], ur[t], ub[t], phi_r[t], phi_b[t], psi_r[t], psi_b[t], theta_r1, theta_r2, theta_b1, theta_b2)
            phi_r[t-1] = phi_r[t] - h * dphi_r
            psi_r[t - 1] = psi_r[t] - h * dpsi_r
            phi_b[t - 1] = phi_b[t] - h * dphi_b
            psi_b[t - 1] = psi_b[t] - h * dpsi_b
            # print("gr(vr)==", gr(vr[t]))
            # print("gb(vb)==", gb(vb[t]))
        else:
            # =========================== adjoint evolution at times other than pulse times ============
            # theta_r1 = theta(kr, pkr, pr[t])
            # theta_r2 = theta(kr, pkr, cr[t])
            # theta_b1 = theta(kb, pkb, pb[t])
            # theta_b2 = theta(kb, pkb, cb[t])
            dphi_r, dpsi_r, dphi_b, dpsi_b = adjBlue(pr[t], pb[t], cr[t], cb[t], ur[t], ub[t], phi_r[t], phi_b[t],
                                                     psi_r[t], psi_b[t], theta_r1, theta_r2, theta_b1, theta_b2)
            phi_r[t - 1] = phi_r[t] - h * dphi_r
            psi_r[t - 1] = psi_r[t] - h * dpsi_r
            phi_b[t - 1] = phi_b[t] - h * dphi_b
            psi_b[t - 1] = psi_b[t] - h * dpsi_b
    return phi_r, psi_r, phi_b, psi_b

def optimalStrategyRed():
    for t in range(t_interval):
        if t!=0 and t!=t_interval-1 and t % pulse_interval_r==0:   # vr
        # if t in pulse_red_lst:
            # =================search optimal impulsive strategy ======================
            b = np.arange(vr_low, vr_upp, step)
            hr2 = [-b[j] + gr(b[j]) * (np.dot(mu_r[t], 1 - cr[t]) - np.dot(lambda_r[t], pr[t])) for j in range(len(b))]
            j = np.argmax(hr2)
            vr[t] = b[j]

            a = np.arange(ur_low, ur_upp, step)  ### paremeter
            hr1 = [np.dot(lambda_b[t] * fb(a[i]), 1 - pb[t] - cb[t]) - a[i] for i in range(len(a))]
            i = np.argmax(hr1)
            ur[t] = a[i]

        else: # ur
            a = np.arange(ur_low, ur_upp, step)  ### paremeter
            hr1 = [np.dot(lambda_b[t] * fb(a[i]), 1-pb[t]-cb[t]) - a[i] for i in range(len(a))]
            # hr1 = [fb(ur[t]) * (np.dot(lambda_b[t], 1 - pb[t]) - np.dot(mu_b[t],cb[t])) - a[i] for i in range(len(a))]
            i = np.argmax(hr1)
            ur[t] = a[i]

    return vr,ur
def optimalStrategyBlue():
    for t in range(t_interval):
        if t!=0 and t!=t_interval-1 and t % pulse_interval_b==0:   # vb
            # =================search optimal impulsive strategy ======================
            b = np.arange(vb_low, vb_upp,step)
            hb2 = [-b[j] + gb(b[j]) * (np.dot(psi_b[t], 1-cb[t]) - np.dot(phi_b[t], pb[t])) for j in range(len(b))]
            j = np.argmax(hb2)
            vb[t] = b[j]

            a = np.arange(ub_low, ub_upp, step)  ### paremeter
            hb1 = [np.dot(phi_r[t] * fr(a[i]), 1 - pr[t] - cr[t]) - a[i] for i in range(len(a))]
            i = np.argmax(hb1)
            ub[t] = a[i]
        else: # ub
            a = np.arange(ub_low, ub_upp, step)  ### paremeter
            hb1 = [np.dot(phi_r[t] * fr(a[i]), 1 - pr[t] - cr[t]) - a[i] for i in range(len(a))]
            # hb1 = [fr(ub[t]) * (np.dot(lambda_r[t], 1 - pr[t]) - np.dot(mu_r[t], cr[t])) - a[i] for i in range(len(a))]
            i = np.argmax(hb1)
            ub[t] = a[i]
    return vb,ub

def payoffRed():
    sum_pb = sum(pb[t].sum(axis=0) for t in range(t_interval))
    sum_cr = sum(cr[t].sum(axis=0) for t in range(t_interval))
    sum_pr = sum(pr[t].sum(axis=0) for t in range(t_interval))
    sum_ur = sum(ur)    # condition is the initialization is zero
    sum_vr = sum(vr)    # condition is the initialization is zero
    jred = (brp * sum_pb + brc * sum_cr - lr * sum_pr - sum_ur) * h - sum_vr   # note this value maybe negative
    return jred

def payoffBlue():
    sum_pr = sum(pr[t].sum(axis=0) for t in range(t_interval))
    sum_cb = sum(cb[t].sum(axis=0) for t in range(t_interval))
    sum_pb = sum(pb[t].sum(axis=0) for t in range(t_interval))
    sum_ub = sum(ub)    # condition is the initialization is zero
    sum_vb = sum(vb)    # condition is the initialization is zero
    # print(sum_pr, sum_cb, sum_pb, sum_ub, sum_vb)
    jblue = (bbp * sum_pr + bbc * sum_cb - lb * sum_pb - sum_ub) * h - sum_vb   # note this value maybe negative

    return jblue





if __name__ == '__main__':
    ######################### need the network function to get k and pk #############################?????????????
    # Create graph on facebook
    fpr = "./data/socfb-Haverford76.mtx"  ## facebook red 1.4k
    fpb = "./data/socfb-Reed98.mtx"  ## facebook blue 1k

    # fpr = "./data/socfb-Bingham82.mtx"   ## facebook red 10k
    # fpb = "./data/socfb-Cal65.mtx"   ## facebook blue 11k

    # fpr = "./data/socfb-UF21.mtx"  ## facebook red 35k
    # fpb = "./data/socfb-OR.mtx"  ## facebook blue 63k

    # fpr = "./data/socfb-A-anon.mtx"  ## facebook red 3M
    # fpb = "./data/socfb-B-anon.mtx"  ## facebook blue 3M

    # Create graph on twitter

    # fpr = "./data/rt_barackobama.edges"  ## twitter  10k
    # fpb = "./data/rt-pol.txt"        ## twitter 18k

    # fpr = "./data/rt_assad.edges"  ## twitter  2k
    # fpb = "./data/rt_voteonedirection.edges"        ## twitter 2k

    # fpr = "./data/rt-higgs.edges"  ## twitter  425k
    # fpb = "./data/rt-retweet-crawl.mtx"  ## twitter 1m

    Gr = graphs(fpr)
    Gb = graphs(fpb)

    kr, pkr = my_degree(Gr)
    kb, pkb = my_degree(Gb)
    avd_r = np.dot(kr, pkr)
    avd_b = np.dot(kb, pkb)

    print("avd_r", avd_r)
    # print("==")
    print("avd_b", avd_b)


    T = 5
    h = 0.001
    t_interval = int(T / h) + 1  # 总刻度， 故下标应该是[0, t_interval-1]
    pulse_interval_r = 500
    pulse_interval_b = 400

    maxiter = 10
    step =0.1

    #
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
    #             for m in range(K - 1):
    #                 a1 += ((t_interval - 1) - (rest[term] - 1)) / (K - 1)
    #                 b.append(int(a1))
    #         c.append(b)
    #     return c

    # K =9
    # pulse_red = pulse_time(700)
    # for i in range(len(pulse_red)):
    #     print("i=", i)
    #     print(pulse_red[i])   ## i=  46-603; 55-652; 64-702; 73-751; 82-801;
    # # pulse_red_lst = pulse_red[12]
    # # print(pulse_red_lst)



  ### transition rates
    # min_val = 1  # 0.001 scaled up by 1000
    # max_val = 9  # 0.09 scaled up by 1000
    # step1 = 1  # 0.002 scaled up by 1000
    # # # Generate the list
    # rates = [i / 100 for i in range(min_val, max_val + 1, step1)] # [0.01, 0.09]
    # rates=[200, 300, 400, 500, 600]
    # rates_lst1 = []
    # rates_lst2 = []


    # upp_r = [2,3,4,5,6,7,8]
    # low_r = [1, 2, 3, 4, 5, 6,7]



    # # min_v = 1  # 0.001 scaled up by 1000
    # # max_v = 10  # 0.09 scaled up by 1000
    # # stepv = 1 # 0.002 scaled up by 1000
    # # # Generate the list
    # # lr_lst = [i / 10 for i in range(min_v, max_v+ 1, stepv)]
    # for i in range(len(low_r)):
    #     print("****i=")
    #     print(low_r[i])

    # pulse_interval_r = rates[i]

    # fb
    beta_r = 0.02
    gamma_r1 = 0.03
    gamma_r2 = 0.01
    delta_r = 0.02

    beta_b = 0.03
    gamma_b1 = 0.04
    gamma_b2 = 0.02
    delta_b = 0.01

    # # ## tw
    # beta_r = 0.03
    # gamma_r1 = 0.04
    # gamma_r2 = 0.02
    # delta_r = 0.02
    #
    # beta_b = 0.04
    # gamma_b1 = 0.05
    # gamma_b2 = 0.03
    # delta_b = 0.01


    c_r = 0.6
    c_b = 0.05
    coeff = 1

    lowr = 1
    uppr = 8
    lowb = 3
    uppb = 10

    brp = brc = lr = bbp = bbc = lb = coeff

    ur_low = lowr
    ur_upp = uppr
    vr_low = lowr
    vr_upp = uppr

    ub_low = lowb
    ub_upp = uppb
    vb_low = lowb
    vb_upp = uppb

    ## initial all variables (states, co_states, controls, payoffs)
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

##### serve for red team
    lambda_r = np.zeros((t_interval, len(kr)))
    mu_r = np.zeros((t_interval, len(kr)))
    lambda_b = np.zeros((t_interval, len(kb)))
    mu_b = np.zeros((t_interval, len(kb)))

##### serve for blue team
    phi_r = np.zeros((t_interval, len(kr)))
    psi_r = np.zeros((t_interval, len(kr)))
    phi_b = np.zeros((t_interval, len(kb)))
    psi_b = np.zeros((t_interval, len(kb)))


    # ur = np.ones(t_interval)
    # ub = np.ones(t_interval)
    # ur = np.random.random(t_interval)
    # ub = np.random.random(t_interval)
    ur = np.zeros(t_interval)
    ub = np.zeros(t_interval)
    vr = np.zeros(t_interval)
    vb = np.zeros(t_interval)

    jr = np.zeros(maxiter + 1)  #[0, maxiter]
    jb = np.zeros(maxiter + 1)  # [0, maxiter]

    pr, cr = forwardRed()
    pb, cb = forwardBlue()

    jr[0] = payoffRed()
    jb[0] = payoffBlue()

    start_time = time.time()
    for iter in range(0, maxiter):  #[0, maxiter)

        print("========================backward=========================")
        lambda_r, mu_r, lambda_b, mu_b = backwardRed()
        phi_r, psi_r, phi_b, psi_b = backwardBlue()
        print('=============================strategy============================')
        vr, ur = optimalStrategyRed()
        vb, ub = optimalStrategyBlue()
        # print("ur1=", ur[:20])
        # print("ur2=", ur[:-20])
        print("========================forward=========================")
        pr, cr = forwardRed()
        pb, cb = forwardBlue()
        print('=============================payoff============================')
        jr[iter + 1] = payoffRed()
        jb[iter + 1] = payoffBlue()
        # print(pr, cr, pb, cb)
        # print("payoffRed==", jr)
        # print("payoffBlue==", jb)

#
#
#     # dpr = pd.DataFrame(pr)
#     # dcr = pd.DataFrame(cr)
#     # dpb = pd.DataFrame(pb)
#     # dcb = pd.DataFrame(cb)
#     # dur = pd.DataFrame(ur)
#     # dvr = pd.DataFrame(vr)
#     # dub = pd.DataFrame(ub)
#     # dvb = pd.DataFrame(vb)
#     djr = pd.DataFrame(jr)
#     djb = pd.DataFrame(jb)
    end_time = time.time()
    print('time consumption:\n',end_time-start_time)
    x = np.arange(maxiter + 1)
    y_r = jr
    y_b = jb
    plt.plot(x, y_r, color='r',label='Jr' )
    plt.plot(x, y_b, color='b',label='Jb')
    plt.legend(loc="lower center")
    plt.xlabel('Iteration')
    plt.ylabel('(Jr, Jb)')
    plt.savefig("./result/figs/converge/con_fb_1k.pdf")
    plt.show()

#
#
#
#     # ### reansition rates  [在迭代外面，for i 里面]
#     # rates_lst1.append(jr[-1])
#     # rates_lst2.append(jb[-1])
#
#
#     # ### save result  # [在for i 外面]
#     a = 'tw_startTime__'
#     b = pulse_red_lst
    # dj = pd.DataFrame({'col1':rates_lst1, 'col2':rates_lst2})

    # dpr.to_csv("./result/result_new33/pr_" + a + str(b) +".csv", index=False)
    # dcr.to_csv("./result/result_new33/cr_" + a + str(b)+".csv", index=False)
    # dpb.to_csv("./result/result_new33/pb_" + a + str(b)+".csv", index=False)
    # dcb.to_csv("./result/result_new33/cb_" + a + str(b)+".csv", index=False)
    # dur.to_csv("./result/result_new33/ur_" + a + str(b)+".csv", index=False)
    # dvr.to_csv("./result/result_new33/vr_" + a + str(b)+".csv", index=False)
    # dub.to_csv("./result/result_new33/ub_" + a + str(b)+".csv", index=False)
    # dvb.to_csv("./result/result_new33/vb_" + a + str(b)+".csv", index=False)
    # djb.to_csv("./result/result_new33/jb_" + a + str(b)+".csv", index=False)
    # djr.to_csv("./result/result_new33/jr_" + a + str(b) + ".csv", index=False)
    # dj.to_csv("./result/result_new33/j_" + a + str(b)+".csv", index=False)


    # ### caculate weighted average of states
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
    # x2 = 1 + np.arange(t_interval)
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    #
    # ax[0].plot(x2, ne_cr, 'g', label='avg_Cr')
    # ax[0].plot(x2, ne_pr, 'm', label='avg_Pr')
    # ax[0].legend(loc="upper right")
    # ax[0].set_ylabel("States for Red Team")
    # ax[0].set_xlabel("Time")
    #
    # ax[1].plot(x2, ne_cb, 'g--', label='avg_Cb')
    # ax[1].plot(x2, ne_pb, 'm--', label='avg_Pb')
    # ax[1].legend(loc="upper right")
    # ax[1].set_ylabel("States for Blue Team")
    # ax[1].set_xlabel("Time")
    #
    #
    # ##
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    #
    # ax[0].plot(x2, ur, 'g', label='ur')
    # ax[0].plot(x2, vr, 'm', label='vr')
    # ax[0].legend(loc="upper right")
    # ax[0].set_ylabel("Controls for Red Team")
    # ax[0].set_xlabel("Time")
    #
    # ax[1].plot(x2, ub, 'g--', label='ub')
    # ax[1].plot(x2, vb, 'm--', label='vb')
    # ax[1].legend(loc="upper right")
    # ax[1].set_ylabel("Controls for Blue Team")
    # ax[1].set_xlabel("Time")


    # plt.show()


