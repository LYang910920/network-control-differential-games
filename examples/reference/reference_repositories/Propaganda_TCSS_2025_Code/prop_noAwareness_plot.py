import numpy as np
from math import *
from prop_network import adj_matrix
import matplotlib.pyplot as plt
import random
import pandas as pd
from decimal import Decimal

# # # ###  exp1: effect of awareness on
# j_ISaSuR_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/J4_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# j_ISaSuR_rt = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/J4_rt_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# j_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/J4_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# #
# j_SIR_fb = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/J4_fb00.002_0_0.001_00.001_0.01.csv")
# j_SIR_rt = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/J4_rt00.002_0_0.001_00.001_0.01.csv")
# j_SIR_yb = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/J4_yb00.002_0_0.001_00.001_0.01.csv")
# # #
# jnew_fb = j_ISaSuR_fb.values[-1][-1]
# jold_fb = j_SIR_fb.values[-1][-1]
# jnew_rt = j_ISaSuR_rt.values[-1][-1]
# jold_rt = j_SIR_rt.values[-1][-1]
# jnew_yb = j_ISaSuR_yb.values[-1][-1]
# jold_yb = j_SIR_yb.values[-1][-1]
#
# fig, axes = plt.subplots(1, 3, figsize=(7.5, 4), sharey=True)
# x = 1 + np.arange(0, 6, 1)
# # axes[0].stem(x[0], jnew_fb, linefmt='C3-', markerfmt='C3o', label='$IS_{a}S_{u}R$')
# # axes[0].stem(x[1], jold_fb, basefmt='C0-', markerfmt='C0o', label='$SIR$')
# axes[0].bar(x[0] - 0.1, jnew_fb, width=0.35, color='red', label='$IS_{a}S_{u}R$')
# axes[0].bar(x[1] + 0.1, jold_fb, width=0.35, color='blue', label='$SIR$')
# axes[0].set_title("On Facebook")
# axes[0].set_xlim([0,3])
# # axes[0].hlines(jnew_fb, 0, x[0], color='C3', linestyles='--')
# # axes[0].hlines(jold_fb, 0, x[1], color='C0', linestyles='--')
# axes[0].set_xlabel("c'", fontweight='bold')
# axes[0].set_ylabel("J(c')", fontweight='bold')
# axes[0].set_xticks(x[0:2], ['$(c^a, c^u)$','$c^{no}$'])
# axes[0].legend(fontsize=4.5, loc="upper right")
#
# # axes[1].stem(x[0], jnew_rt, linefmt='C3-', markerfmt='C3o', label='$IS_{a}S_{u}R$')
# # axes[1].stem(x[1], jold_rt, basefmt='C0-', markerfmt='C0o', label='$SIR$')
# axes[1].bar(x[0] - 0.1, jnew_rt, width=0.35, color='red', label='$IS_{a}S_{u}R$')
# axes[1].bar(x[1] + 0.1, jold_rt, width=0.35, color='blue', label='$SIR$')
# axes[1].set_title("On Twitter")
# axes[1].set_xlim([0,3])
# # axes[1].hlines(jnew_rt, 0, x[0], color='C3', linestyles='--')
# # axes[1].hlines(jold_rt, 0, x[1], color='C0', linestyles='--')
# axes[1].set_xlabel("c'", fontweight='bold')
# axes[1].set_xticks(x[0:2], ['$(c^a, c^u)$','$c^{no}$'])
# axes[1].legend(fontsize=4.5, loc="upper right")
# #
# # ########################################
# # axes[2].stem(x[0], jnew_yb, linefmt='C3-', markerfmt='C3o', label='$IS_{a}S_{u}R$')
# # axes[2].stem(x[1], jold_yb, basefmt='C0-', markerfmt='C0o', label='$SIR$')
# axes[2].bar(x[0] - 0.1, jnew_yb, width=0.35, color='red', label='$IS_{a}S_{u}R$')
# axes[2].bar(x[1] + 0.1, jold_yb, width=0.35, color='blue', label='$SIR$')
# axes[2].set_title("On YouTube")
# axes[2].set_xlim([0,3])
# # axes[2].hlines(jnew_yb, 0, x[0], color='C3', linestyles='--')
# # axes[2].hlines(jold_yb, 0, x[1], color='C0', linestyles='--')
# axes[2].set_xlabel("c'", fontweight='bold')
# axes[2].set_xticks(x[0:2], ['$(c^a, c^u)$','$c^{no}$'])
# axes[2].legend(fontsize=4.5, loc="upper right")
#
# # plt.show()
# # #
# plt.savefig("./data/exp1-2_fig/awareness_comparison_bar.pdf")

#
#
# # # #### exp 2: effect of awareness on dynamics of R state
# r_ISaSuR_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r4_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# r_SIR_fb = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/r4_fb00.002_0_0.001_00.001_0.01.csv")
# r_ISaSuR_rt = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r4_rt_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# r_SIR_rt = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/r4_rt00.002_0_0.001_00.001_0.01.csv")
# r_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r4_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# r_SIR_yb = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/r4_yb00.002_0_0.001_00.001_0.01.csv")
#
#
# # r_ISaSuR_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su4_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# # r_SIR_fb = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/su4_fb00.002_0_0.001_00.001_0.01.csv")
# # r_ISaSuR_rt = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su4_rt_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# # r_SIR_rt = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/su4_rt00.002_0_0.001_00.001_0.01.csv")
# # r_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su4_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv")
# # r_SIR_yb = pd.read_csv("./data/exp_data/toUse/SIR_lessNodes/su4_yb00.002_0_0.001_00.001_0.01.csv")
#
#
# rnew_fb = r_ISaSuR_fb.values
# rold_fb = r_SIR_fb.values
# rnew_rt = r_ISaSuR_rt.values
# rold_rt = r_SIR_rt.values
# rnew_yb = r_ISaSuR_yb.values
# rold_yb = r_SIR_yb.values
# #
# rnew_fb_avg = []
# rold_fb_avg = []
# rnew_rt_avg = []
# rold_rt_avg = []
# rnew_yb_avg = []
# rold_yb_avg = []
# #
# # #### average R at time t in network 1
# for row2 in rnew_fb:
#     avg_w = sum(row2) / len(row2)
#     rnew_fb_avg.append(avg_w)
# for row1 in rold_fb:
#     avg_wout = sum(row1)/len(row1)
#     rold_fb_avg.append(avg_wout)
# #
# for row2 in rnew_rt:
#     avg_w = sum(row2) / len(row2)
#     rnew_rt_avg.append(avg_w)
# for row1 in rold_rt:
#     avg_wout = sum(row1)/len(row1)
#     rold_rt_avg.append(avg_wout)
# #
# for row2 in rnew_yb:
#     avg_w = sum(row2) / len(row2)
#     rnew_yb_avg.append(avg_w)
# for row1 in rold_yb:
#     avg_wout = sum(row1)/len(row1)
#     rold_yb_avg.append(avg_wout)
#
#
# fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.7), sharey=True)
# x = 1 + np.arange(len(rnew_rt_avg))  # x-axis
#
# rw_avg, = axes[0].plot(x, rnew_fb_avg, label="ISaSuR", linestyle="-.")
# rwout_avg, = axes[0].plot(x, rold_fb_avg, label="SIR", linestyle="-")
# axes[0].set_title("On Facebook")
# axes[0].set_ylabel('$\overline{R}(t)$')
# axes[0].set_xlabel('t')
# axes[0].legend(fontsize=6, handles=[rw_avg, rwout_avg], loc="lower center")
# axes[0].set_xticks([x[0], x[250], x[500]], ['0','2.5', '5'])
# #
# #
# rnew_rt_a, = axes[1].plot(x, rnew_rt_avg, label="ISaSuR", linestyle="-.")
# rold_rt_a, = axes[1].plot(x, rold_rt_avg, label="SIR", linestyle="-")
# axes[1].set_title("On Twitter")
# axes[1].set_xlabel('t')
# axes[1].legend(fontsize=6, handles=[rnew_rt_a, rold_rt_a], loc="lower center")
# axes[1].set_xticks([x[0], x[250], x[500]], ['0','2.5', '5'])
#
# rnew_yb_a, = axes[2].plot(x, rnew_yb_avg, label="ISaSuR", linestyle="-.")
# rold_yb_a, = axes[2].plot(x, rold_yb_avg, label="SIR", linestyle="-")
# axes[2].set_title("On YouTube")
# axes[2].set_xlabel('t')
# axes[2].legend(fontsize=6, handles=[rnew_yb_a, rold_yb_a], loc="lower center")
# axes[2].set_xticks([x[0], x[200], x[500]], ['0','2', '5'])
#
# plt.show()
# # #
# plt.savefig("./data/exp1-2_fig/awareness_dynamics_4.pdf")
# # plt.savefig("./data/exp1-2_fig/awareness_dynamics.eps")



# added experiments 1  描绘网络状态在3不同的网络上
#
# # r_ISaSuR_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r4_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# # r_ISaSuR_tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r4_rt_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# # r_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r4_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
#
#

# sa_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/sa_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# su_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# r_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values


# r_low_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/ran_r_low_fb.csv").values
# r_mid_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/ran_r_mid_fb.csv").values
# r_upp_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/ran_r_upp_fb.csv").values
#
# r_low_tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/ran_r_low_tw.csv").values
# r_mid_tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/ran_r_mid_tw.csv").values
# r_upp_tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/ran_r_upp_tw.csv").values
#
# r_low_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/fixed_r_low_yb.csv").values
# r_mid_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/fixed_r_mid_yb.csv").values
# r_upp_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_heuristic/fixed_r_upp_yb.csv").values
#
# #
#
#
# r_ISaSuR_fb_avg = []
# r_ISaSuR_tw_avg = []
# r_ISaSuR_yb_avg = []
# sa_ISaSuR_yb_avg = []
# su_ISaSuR_yb_avg = []
#
# r_low_fb_avg = []
# r_mid_fb_avg = []
# r_upp_fb_avg = []
#
#
# r_low_tw_avg = []
# r_mid_tw_avg = []
# r_upp_tw_avg = []
#
# r_low_yb_avg = []
# r_mid_yb_avg = []
# r_upp_yb_avg = []
#
#
# # # #### average R at time t in network 1
# for row2 in r_ISaSuR_fb:  #(t,node)
#     avg_w = sum(row2) / len(row2)
#     r_ISaSuR_fb_avg.append(avg_w)
#
# for row1 in r_ISaSuR_tw:
#     avg_wout = sum(row1)/len(row1)
#     r_ISaSuR_tw_avg.append(avg_wout)
# #
# for row2 in r_ISaSuR_yb:
#     avg_w = sum(row2) / len(row2)
#     r_ISaSuR_yb_avg.append(avg_w)
# for row2 in sa_ISaSuR_yb:
#     avg_w = sum(row2) / len(row2)
#     sa_ISaSuR_yb_avg.append(avg_w)
# for row2 in su_ISaSuR_yb:
#     avg_w = sum(row2) / len(row2)
#     su_ISaSuR_yb_avg.append(avg_w)
#
# # fb
# for row1 in r_low_fb:
#     avg_wout = sum(row1)/len(row1)
#     r_low_fb_avg.append(avg_wout)
#
# for row2 in r_mid_fb:
#     avg_w = sum(row2) / len(row2)
#     r_mid_fb_avg.append(avg_w)
#
# for row1 in r_upp_fb:
#     avg_wout = sum(row1)/len(row1)
#     r_upp_fb_avg.append(avg_wout)
#
#
# # tw
#
# for row1 in r_low_tw:
#     avg_wout = sum(row1) / len(row1)
#     r_low_tw_avg.append(avg_wout)
#
# for row2 in r_mid_tw:
#     avg_w = sum(row2) / len(row2)
#     r_mid_tw_avg.append(avg_w)
#
# for row1 in r_upp_tw:
#     avg_wout = sum(row1) / len(row1)
#     r_upp_tw_avg.append(avg_wout)
#
#
#  # yb
# for row1 in r_low_yb:
#     avg_wout = sum(row1) / len(row1)
#     r_low_yb_avg.append(avg_wout)
#
# for row2 in r_mid_yb:
#     avg_w = sum(row2) / len(row2)
#     r_mid_yb_avg.append(avg_w)
#
# for row1 in r_upp_yb:
#     avg_wout = sum(row1) / len(row1)
#     r_upp_yb_avg.append(avg_wout)
#
# #
# fig, axes = plt.subplots(1, 2, figsize=(15, 3.7), sharey=True)
# x = 1 + np.arange(len(r_ISaSuR_yb_avg))  # x-axis
#
# # fb
# r_ISaSuR_fb_ = axes[0].plot(x, r_ISaSuR_fb_avg,'r-.', label="$c^*$")
# r_low_fb_ = axes[0].plot(x, r_low_fb_avg, 'g-', label="$c_l$")
# r_mid_fb_ = axes[0].plot(x, r_mid_fb_avg, 'y--', label="$c_m$")
# r_upp_fb_ = axes[0].plot(x, r_upp_fb_avg, 'm:', label="$c_h$")
# axes[0].set_title("On Facebook")
# axes[0].set_ylabel('$\overline{R}(t)$')
# axes[0].set_xlabel('t')
# axes[0].legend(fontsize=6, loc="upper left")
# axes[0].set_xticks([x[0], x[250], x[500]], ['0','2.5', '5'])
# # tw
# r_ISaSuR_tw_ = axes[1].plot(x, r_ISaSuR_tw_avg, 'r-.', label="$c^*$")
# r_low_tw_ = axes[1].plot(x, r_low_tw_avg, 'g-', label="$c_l$")
# r_mid_tw_ = axes[1].plot(x, r_mid_tw_avg, 'y--', label="$c_m$")
# r_upp_tw_ = axes[1].plot(x, r_upp_tw_avg, 'm:', label="$c_h$")
# axes[1].set_title("On Twitter")
# axes[1].set_xlabel('t')
# axes[1].legend(fontsize=6, loc="upper left")
# axes[1].set_xticks([x[0], x[250], x[500]], ['0','2.5', '5'])
# #yb
# r_ISaSuR_yb_ = axes[2].plot(x, r_ISaSuR_yb_avg, 'r-.', label="$c^*$")
# r_low_yb_ = axes[2].plot(x, r_low_yb_avg, 'g-', label="$c_l$")
# r_mid_yb_ = axes[2].plot(x, r_mid_yb_avg, 'y--', label="$c_m$")
# r_upp_yb_ = axes[2].plot(x, r_upp_yb_avg, 'm:', label="$c_h$")
# axes[2].set_title("On YouTube")
# axes[2].set_xlabel('t')
# axes[2].legend(fontsize=6, loc="upper left")
# axes[2].set_xticks([x[0], x[250], x[500]], ['0','2.5', '5'])

# #yb, sa, su, r, i

# Assuming these variables are lists
# i_ISaSuR_yb_avg = [float(Decimal(1) - Decimal(r) - Decimal(sa) - Decimal(su)) for r, sa, su in zip(r_ISaSuR_yb_avg, sa_ISaSuR_yb_avg, su_ISaSuR_yb_avg)]

# print(r_ISaSuR_yb_avg[:5])
# print(sa_ISaSuR_yb_avg[:5])
# print(su_ISaSuR_yb_avg[:5])
# r_ISaSuR_yb_ = axes[2].plot(x, r_ISaSuR_yb_avg, 'r-.', label="$\overline{R}(t)$")
# i_ISaSuR_yb_ = axes[2].plot(x, i_ISaSuR_yb_avg, 'g-', label="$\overline{I}(t)$")
# sa_ISaSuR_yb_ = axes[2].plot(x, sa_ISaSuR_yb_avg, 'y--', label="$\overline{Sa}(t)$")
# su_ISaSuR_yb_ = axes[2].plot(x, su_ISaSuR_yb_avg, 'm:', label="$\overline{Su}(t)$")
# axes[2].set_title("On YouTube")
# axes[2].set_xlabel('t')
# axes[2].legend(fontsize=6, loc="upper left")
# axes[2].set_xticks([x[0], x[250], x[500]], ['0','2.5', '5'])

# r_ISaSuR_yb_ = axes[1].plot(x, r_ISaSuR_yb_avg, 'r-.', label="$\overline{R}(t)$")
# i_ISaSuR_yb_ = axes[1].plot(x, i_ISaSuR_yb_avg, 'g-', label="$\overline{I}(t)$")
# sa_ISaSuR_yb_ = axes[1].plot(x, sa_ISaSuR_yb_avg, 'y--', label="$\overline{Sa}(t)$")
# su_ISaSuR_yb_ = axes[1].plot(x, su_ISaSuR_yb_avg, 'm:', label="$\overline{Su}(t)$")
# axes[1].set_title("On YouTube")
# axes[1].set_xlabel('t')
# axes[1].legend(fontsize=6, loc="upper left")
# axes[1].set_xticks([x[0], x[250], x[500]], ['0','2.5', '5'])


# plt.show()


####### 描绘所有网络的STATE, Sa, Su, R, I

# r
# fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r_fixed_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values

# sa
# fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/sa_fixed_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/sa_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/sa_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values

# su
fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su_fixed_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values


# # # i
# r_ISaSuR_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r_fixed_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# sa_ISaSuR_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/sa_fixed_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# su_ISaSuR_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su_fixed_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
#
# r_ISaSuR_fb_dec = np.vectorize(Decimal)(r_ISaSuR_fb)
# sa_ISaSuR_fb_dec = np.vectorize(Decimal)(sa_ISaSuR_fb)
# su_ISaSuR_fb_dec = np.vectorize(Decimal)(su_ISaSuR_fb)
# # 逐元素计算 i
# fb = 1 - r_ISaSuR_fb_dec - sa_ISaSuR_fb_dec - su_ISaSuR_fb_dec
#
# # # 如果需要将结果转换回浮点数
# # fb = np.vectorize(float)(i_fb)
#
#
# # # states for tw
# r_ISaSuR_tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# sa_ISaSuR_tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/sa_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# su_ISaSuR_tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# r_ISaSuR_tw_dec = np.vectorize(Decimal)(r_ISaSuR_tw)
# sa_ISaSuR_tw_dec = np.vectorize(Decimal)(sa_ISaSuR_tw)
# su_ISaSuR_tw_dec = np.vectorize(Decimal)(su_ISaSuR_tw)
# # 逐元素计算 i
# tw = 1 - r_ISaSuR_tw_dec - sa_ISaSuR_tw_dec - su_ISaSuR_tw_dec
#
# #
# # # states for yb
# r_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/r_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# sa_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/sa_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# su_ISaSuR_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/su_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values
# r_ISaSuR_yb_dec = np.vectorize(Decimal)(r_ISaSuR_yb)
# sa_ISaSuR_yb_dec = np.vectorize(Decimal)(sa_ISaSuR_yb)
# su_ISaSuR_yb_dec = np.vectorize(Decimal)(su_ISaSuR_yb)
# # 逐元素计算 i
# yb = 1 - r_ISaSuR_yb_dec - sa_ISaSuR_yb_dec - su_ISaSuR_yb_dec

#
# caculate avgs for all states
ISaSuR_fb_avg = []
ISaSuR_tw_avg = []
ISaSuR_yb_avg = []

# caculate avg for each state
for row2 in fb:
    avg_w = sum(row2) / len(row2)
    ISaSuR_fb_avg.append(avg_w)
for row2 in tw:
    avg_w = sum(row2) / len(row2)
    ISaSuR_tw_avg.append(avg_w)
for row2 in yb:
    avg_w = sum(row2) / len(row2)
    ISaSuR_yb_avg.append(avg_w)

x = 1 + np.arange(len(ISaSuR_fb_avg))  # x-axis
# 创建图形
plt.figure(figsize=(10,5))


plt.plot(x, ISaSuR_fb_avg, color='red', linestyle='-.', linewidth=3, label="Facebook")
plt.plot(x, ISaSuR_tw_avg, color='green', linestyle='--', linewidth=3, label="Twitter")
plt.plot(x, ISaSuR_yb_avg, color='blue', linestyle=':', linewidth=3, label="YouTube")


# 设置标题和标签
# plt.title("变化趋势图", fontsize=14)
plt.xticks([x[0], x[250], x[500]], ['0','2.5', '5'],fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel("Time", fontsize=18)
plt.ylabel("$\\overline{S^u}(t)$", fontsize=18)

# 设置图例
plt.legend(fontsize=14, loc="lower left", frameon=True,handlelength=4.2, borderpad=1.0, labelspacing=1.2, edgecolor='black')

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
# plt.show()
plt.savefig("./data/exp5_fig/su_avgs.pdf")


# #### added experiments 2
#
# # # Our J
# #
# #
# j_fb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/J_fixed_fb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values[-1][-1]
# j_tw = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/J_fixed_tw_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values[-1][-1]
# j_yb = pd.read_csv("./data/exp_data/toUse/ISaSuR_lessNodes/J_fixed_yb_0.0010.002_0.001_0.001_0.0020.001_0.01.csv").values[-1][-1]
#
#
# # # FB
# fb_2Ran = pd.read_csv("./data/exp_data/random/fixed_J_fb_2Ran_mixed.csv").values[-1][-1]
# fb_caRan = pd.read_csv("./data/exp_data/random/fixed_J_fb_caRan_mixed.csv").values[-1][-1]
# fb_cuRan = pd.read_csv("./data/exp_data/random/fixed_J_fb_cuRan_mixed.csv").values[-1][-1]
#
# # # TW
# tw_2Ran = pd.read_csv("./data/exp_data/random/fixed_J_tw_2Ran_mixed.csv").values[-1][-1]
# tw_caRan = pd.read_csv("./data/exp_data/random/fixed_J_tw_caRan_mixed.csv").values[-1][-1]
# tw_cuRan = pd.read_csv("./data/exp_data/random/fixed_J_tw_cuRan_mixed.csv").values[-1][-1]
# #
# # # # YB
# yb_2Ran = pd.read_csv("./data/exp_data/random/fixed_J_yb_2Ran_mixed.csv").values[-1][-1]
# yb_caRan = pd.read_csv("./data/exp_data/random/fixed_J_yb_caRan_mixed.csv").values[-1][-1]
# yb_cuRan = pd.read_csv("./data/exp_data/random/fixed_J_yb_cuRan_mixed.csv").values[-1][-1]
                #此次实验没有用到以下代码
                        # fig, axes = plt.subplots(1, 3, figsize=(13, 8), sharey=True)
                        # x = 1 + np.arange(0, 5, 1)
                        # print(x)
                        # axes[0].bar(x[0] - 0.1, j_fb, width=0.35, color='red', label='$J(c^{a*}, c^{u*})$')
                        # axes[0].bar(x[1] , fb_caRan, width=0.35, color='green', label='$J(c^a_i,c^{u*})$')
                        # axes[0].bar(x[2] + 0.1, fb_cuRan, width=0.35, color='orange', label='$J(c^{a*}, c^u_i)$')
                        # axes[0].bar(x[3] + 2*0.1, fb_2Ran, width=0.35, color='purple', label='$J(c^a_i, c^u_i)$')
                        # axes[0].set_title("On Facebook")
                        # axes[0].set_xlim([0,len(x)])
                        # axes[0].set_xlabel("c'", fontweight='bold')
                        # axes[0].set_ylabel("J(c')", fontweight='bold')
                        # axes[0].set_xticks(x, ['$c^*$','$c_l$', '$c_m$', '$c_h$', ''])
                        # axes[0].legend(fontsize=4.5, loc="upper right")
                        #
                        # axes[1].bar(x[0] - 0.1, j_tw, width=0.35, color='red', label='$J(c^{a*}, c^{u*})$')
                        # axes[1].bar(x[1], tw_caRan, width=0.35, color='green', label='$J(c^a_i,c^{u*})$')
                        # axes[1].bar(x[2] + 0.1, tw_cuRan, width=0.35, color='orange', label='$J(c^{a*}, c^u_i)$')
                        # axes[1].bar(x[3] + 2*0.1, tw_2Ran, width=0.35, color='purple', label='$J(c^a_i, c^u_i)$')
                        # axes[1].set_title("On Twitter")
                        # axes[1].set_xlim([0,len(x)])
                        # axes[1].set_xlabel("c'", fontweight='bold')
                        # axes[1].set_xticks(x, ['$c^*$','$c_l$', '$c_m$', '$c_h$', ''])
                        # axes[1].legend(fontsize=4.5, loc="upper right")
                        #
                        # axes[2].bar(x[0] - 0.1, j_yb, width=0.35, color='red', label='$J(c^{a*}, c^{u*})$')
                        # axes[2].bar(x[1], yb_caRan, width=0.35, color='green', label='$J(c^a_i,c^{u*})$')
                        # axes[2].bar(x[2] + 0.1, yb_cuRan, width=0.35, color='orange', label='$J(c^{a*}, c^u_i)$')
                        # axes[2].bar(x[3] + 2 * 0.1, yb_2Ran, width=0.35, color='purple', label='$J(c^a_i, c^u_i)$')
                        # axes[2].set_title("On YouTube")
                        # axes[2].set_xlim([0,len(x)])
                        # axes[2].set_xlabel("c'", fontweight='bold')
                        # axes[2].set_xticks(x, ['$(c^{a*},c^{u*})$','$(c^a_i,c^{u*})$', '$(c^{a*},c^u_i)$', '$(c^a_i,c^u_i)$', ''], rotation=25,
                        #     ha='center',
                        #     fontsize=12)
                        # axes[2].legend(fontsize=10, loc="upper right",frameon=True,handlelength=1.2, borderpad=0.1, labelspacing=0.1, edgecolor='black')

#
# ###################################### split all subplots############################
# ###################################### split all subplots############################

# import matplotlib.pyplot as plt
# import numpy as np
#
# # 示例数据
# x = 1 + np.arange(0, 5, 1)
#
# # 绘制第一个图 (Facebook)
# fig1, ax1 = plt.subplots(figsize=(10, 8))
# ax1.bar(x[0] - 0.1, j_fb, width=0.35, color='red', label='$J(c^{a*}, c^{u*})$')
# ax1.bar(x[1], fb_caRan, width=0.35, color='green', label='$J(c^a_i,c^{u*})$')
# ax1.bar(x[2] + 0.1, fb_cuRan, width=0.35, color='orange', label='$J(c^{a*}, c^u_i)$')
# ax1.bar(x[3] + 2 * 0.1, fb_2Ran, width=0.35, color='purple', label='$J(c^a_i, c^u_i)$')
# ax1.set_title("On Facebook")
# ax1.set_ylabel("J(c')", fontweight='bold',fontsize=18)
# # 只调整 y 轴刻度标签的字体大小
# plt.setp(ax1.get_yticklabels(), fontsize=18)
# ax1.set_xlim([0, len(x)])
# ax1.set_xlabel("c'", fontweight='bold', fontsize=18)
# ax1.set_ylabel("J(c')", fontweight='bold',fontsize=18)
# ax1.set_yticks([0, 10, 15, 20, 25])  # 示例：根据需要设置 y 轴刻度的位置
# ax1.set_yticklabels([0, 10, 20, 25, 28], fontsize=20)  # 设置 y 轴刻度标签和字体大小
# ax1.set_xticks(x)
# ax1.set_xticklabels(['$(c^{a*},c^{u*})$', '$(c^a_i,c^{u*})$', '$(c^{a*},c^u_i)$', '$(c^a_i,c^u_i)$', ''],
#                     rotation=25, ha='center', fontsize=20)
# ax1.legend(fontsize=20, loc="upper right", frameon=True, handlelength=1.2, borderpad=0.1, labelspacing=0.1, edgecolor='black')
# plt.tight_layout()
# # plt.show()
# plt.savefig("./data/exp5_fig/profit_mixed_fb.pdf")
#
# # 绘制第二个图 (Twitter)
# fig2, ax2 = plt.subplots(figsize=(10, 8))
# ax2.bar(x[0] - 0.1, j_tw, width=0.35, color='red', label='$J(c^{a*}, c^{u*})$')
# ax2.bar(x[1], tw_caRan, width=0.35, color='green', label='$J(c^a_i,c^{u*})$')
# ax2.bar(x[2] + 0.1, tw_cuRan, width=0.35, color='orange', label='$J(c^{a*}, c^u_i)$')
# ax2.bar(x[3] + 2 * 0.1, tw_2Ran, width=0.35, color='purple', label='$J(c^a_i, c^u_i)$')
# ax2.set_title("On Twitter")
# ax2.set_ylabel("J(c')", fontweight='bold',fontsize=18)
# ax2.set_yticks([0, 5, 10, 15, 20, 25, 30])  # 示例：根据需要设置 y 轴刻度的位置
# ax2.set_yticklabels([0, 5, 10, 15, 20, 25, 30], fontsize=20)  # 设置 y 轴刻度标签和字体大小
# ax2.set_xlim([0, len(x)])
# ax2.set_xlabel("c'", fontweight='bold', fontsize=18)
# ax2.set_xticks(x)
# ax2.set_xticklabels(['$(c^{a*},c^{u*})$', '$(c^a_i,c^{u*})$', '$(c^{a*},c^u_i)$', '$(c^a_i,c^u_i)$', ''],
#                     rotation=25, ha='center', fontsize=20)
# ax2.legend(fontsize=20, loc="upper right", frameon=True, handlelength=1.2, borderpad=0.08, labelspacing=0.08, edgecolor='black')
# plt.tight_layout()
# # plt.show()
# plt.savefig("./data/exp5_fig/profit_mixed_tw.pdf")
#
# # 绘制第三个图 (YouTube)
# fig3, ax3 = plt.subplots(figsize=(10, 8))
# ax3.bar(x[0] - 0.1, j_yb, width=0.35, color='red', label='$J(c^{a*}, c^{u*})$')
# ax3.bar(x[1], yb_caRan, width=0.35, color='green', label='$J(c^a_i,c^{u*})$')
# ax3.bar(x[2] + 0.1, yb_cuRan, width=0.35, color='orange', label='$J(c^{a*}, c^u_i)$')
# ax3.bar(x[3] + 2 * 0.1, yb_2Ran, width=0.35, color='purple', label='$J(c^a_i, c^u_i)$')
# ax3.set_title("On YouTube")
# ax3.set_ylabel("J(c')", fontweight='bold',fontsize=18)
# # 只调整 y 轴刻度标签的字体大小
# plt.setp(ax3.get_yticklabels(), fontsize=20)
# ax3.set_xlim([0, len(x)])
# ax3.set_xlabel("c'", fontweight='bold', fontsize=18)
# ax3.set_xticks(x)
# ax3.set_xticklabels(['$(c^{a*},c^{u*})$', '$(c^a_i,c^{u*})$', '$(c^{a*},c^u_i)$', '$(c^a_i,c^u_i)$', ''],
#                     rotation=25, ha='center', fontsize=20)
# ax3.legend(fontsize=20, loc="upper right", frameon=True, handlelength=1.2, borderpad=0.1, labelspacing=0.1, edgecolor='black')
# plt.tight_layout()
# # plt.show()
#
# plt.savefig("./data/exp5_fig/profit_mixed_yb.pdf")