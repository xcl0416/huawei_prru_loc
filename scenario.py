import numpy as np
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from tqdm import tqdm

params = {'anchor_num': 20,               # BS的总个数
          'agent_num': 200,                # UE的总个数
          'known_anchor_num': 2,         # 位置已知的BS的个数
          'dis_com': 20,                 # 通信距离
          'scn_len': 50,                 # 场景大小
          'station_dis_max': 20,         # 最大站间距
          'station_dis_min': 10          # 最小站间距
          }

anchor = np.random.uniform(0, params['scn_len'], (params['anchor_num'], 2 * 1))
agent = np.random.uniform(0, params['scn_len'], (params['agent_num'], 2 * 1))

# 撒点
for i in range(params['known_anchor_num']):
    while True:
        anchor[i] = np.random.uniform(0, params['scn_len'], (1, 2 * 1))
        list_tmp = []
        for j in range(i):
            list_tmp.append(np.linalg.norm(anchor[j] - anchor[i]))
        if [d <= params['dis_com'] * 2 for d in list_tmp].count(True) == 0:
            break
for i in range(params['known_anchor_num'], params['anchor_num']):
    while True:
        anchor[i] = np.random.uniform(0, params['scn_len'], (1, 2 * 1))
        list_tmp = []
        for j in range(i):
            list_tmp.append(np.linalg.norm(anchor[j] - anchor[i]))
        if [d <= params['station_dis_min'] for d in list_tmp].count(True) == 0:
            break

dot1 = plot(agent[:, 0], agent[:, 1], 'r*', label=u'UE')
dot2 = plot(anchor[2:, 0], anchor[2:, 1], 'bo', label=u'BS(unknown)')
dot3 = plot(anchor[:2, 0], anchor[:2, 1], 'k^', label=u'BS(known)')
plt.legend()
plt.show()
a = 1
