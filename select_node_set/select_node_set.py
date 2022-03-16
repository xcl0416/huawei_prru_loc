import numpy as np
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
import math
import random
import copy

params = {'anchor_num': 20,
          'agent_num': 500,
          'known_anchor_num': 2,
          'com_dis': 20,
          'scn_len': 50,
          'station_dis_max': 20,
          'station_dis_min': 10
          }


anchor = np.random.uniform(0, params['scn_len'], (params['anchor_num'], 2 * 1))
agent = np.random.uniform(0, params['scn_len'], (params['agent_num'], 2 * 1))

# 按照站间距要求撒点
for i in range(0, params['anchor_num']):
    while True:
        anchor[i] = np.random.uniform(0, params['scn_len'], (1, 2 * 1))
        list_tmp = []
        for j in range(i):
            list_tmp.append(np.linalg.norm(anchor[j] - anchor[i]))
        if [d <= params['station_dis_min'] for d in list_tmp].count(True) == 0:
            break

list_anchor = []
for i in range(anchor.shape[0]):
    list_anchor.append(np.linalg.norm(anchor[i] - anchor[0]))
index_anchor = sorted(range(len(list_anchor)), key=lambda k: list_anchor[k])
anchor = anchor[index_anchor]

dis_an_ag = np.ones((params['anchor_num'], params['agent_num'])) * 1e10
dis_an_ag4 = np.ones((params['anchor_num'], params['agent_num'])) * 1e10

index_an_ag4 = np.zeros((0, 4))
an_ag = []

# 生成UE和BS的距离信息表
for i in range(anchor.shape[0]):
    for j in range(agent.shape[0]):
        dis = np.linalg.norm(anchor[i] - agent[j])
        if dis <= params['com_dis']:
            dis_an_ag[i][j] = dis

# 找出和UE有距离信息的BS的索引
index_an_ag4 = []
for i in range(dis_an_ag.shape[1]):
    index_4 = list(map(list(dis_an_ag[:, i]).index, heapq.nsmallest(4, list(dis_an_ag[:, i]))))
    index_4_ = []
    for j in range(len(index_4)):
        if dis_an_ag[index_4[j], i] != 1e10:
            index_4_.append(index_4[j])
    index_an_ag4.append(index_4_)
    dis_an_ag4[index_4_, i] = dis_an_ag[index_4_, i]

# np.savetxt('anchor.txt', anchor, delimiter=',', fmt='%.4f')
# np.savetxt('agent.txt', agent, delimiter=',', fmt='%.4f')
# agent_true = np.loadtxt('agent_save_true.txt', delimiter=',', dtype=float)

# 随机选取两个BS，作为已知BS
index_2_known_anchor = random.sample(range(0, params['anchor_num']), 2)

# 有共享UE时的选点
cand_index = set(index_2_known_anchor)
for i in range(params['agent_num']):
    if index_2_known_anchor[0] in index_an_ag4[i] and index_2_known_anchor[1] in index_an_ag4[i]:
        cand_index = cand_index | set(index_an_ag4[i])
final_index = cand_index

# 无共享UE时的选点
if len(final_index) == 2:
    cand_index1 = set([index_2_known_anchor[0]])
    cand_index2 = set([index_2_known_anchor[1]])
    # 有重合BS时的选点
    for i in range(params['agent_num']):
        if index_2_known_anchor[0] in index_an_ag4[i]:
            cand_index1 = cand_index1 | set(index_an_ag4[i])
        if index_2_known_anchor[1] in index_an_ag4[i]:
            cand_index2 = cand_index2 | set(index_an_ag4[i])
    final_index = cand_index1 & cand_index2 | set(index_2_known_anchor)

    # 无重合BS时的选点
    if len(final_index) == 2:
        cand_index11 = copy.deepcopy(cand_index1)
        cand_index22 = copy.deepcopy(cand_index2)
        for i in range(len(cand_index1)):
            for j in range(params['agent_num']):
                if list(cand_index1)[i] in index_an_ag4[j]:
                    cand_index11 = cand_index11 | set(index_an_ag4[j])
        final_index_ = cand_index11 & cand_index2
        final_index = copy.deepcopy(final_index_)

        num_all = 0
        for i in range(len(final_index)):
            set_index_tmp = set([list(final_index)[i]])
            for j in range(params['agent_num']):
                if list(final_index)[i] in index_an_ag4[j]:
                    set_index_tmp = set_index_tmp | set(index_an_ag4[j])
                    # final_index = final_index | set(index_an_ag4[j])
            num_tmp = len(set_index_tmp)
            if num_tmp > num_all:
                num_all = num_tmp
                set_index_tmp_f = set_index_tmp
                med_point = list(final_index)[i]
        # final_index = (final_index & cand_index1)
        set_index_tmp_f_ = copy.deepcopy(set_index_tmp_f)
        set_index_tmp_f = (set_index_tmp_f & cand_index1) | set([med_point])
        final_index = set_index_tmp_f | set(index_2_known_anchor)

# b = index_2_known_anchor
final_index = final_index - set(index_2_known_anchor)
final_index = list(final_index)
index = np.array([index_2_known_anchor + final_index])
np.savetxt('index.txt', index, delimiter=',', fmt='%d')

for i in range(len(final_index)):
    final_index[i] = int(final_index[i])
# dot7 = plot(agent[:, 0], agent[:, 1], 'r*', label=u'UE')
dot7 = plot(anchor[:, 0], anchor[:, 1], 'r*', label=u'UE')

dot8 = plot(anchor[index_2_known_anchor, 0], anchor[index_2_known_anchor, 1], 'bo', label=u'BS(known)')
dot9 = plot(anchor[final_index, 0], anchor[final_index, 1], 'k^', label=u'BS(unknown)')
# plt.legend()
plt.show()
