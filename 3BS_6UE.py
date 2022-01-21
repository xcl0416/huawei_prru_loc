import numpy as np
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from tqdm import tqdm

params = {'anchor_num': 2,               # BS的总个数
          'agent_num': 3,                # UE的总个数
          'known_anchor_num': 2,         # 位置已知的BS的个数
          'dis_com': 20,                 # 通信距离
          'scn_len': 50,                 # 场景大小
          'station_dis_max': 4,         # 最大站间距
          'station_dis_min': 2,          # 最小站间距
          'grid_size': 1
          }

# anchor = np.random.uniform(0, params['scn_len'], (params['anchor_num'], 2 * 1))
# agent = np.random.uniform(0, params['scn_len'], (params['agent_num'], 2 * 1))
# anchor1_dis = np.zeros((20, 1))
# anchor2_dis = np.zeros((20, 1))
# for i in range(params['known_anchor_num']):
#     while True:
#         anchor[i] = np.random.uniform(0, params['scn_len'], (1, 2 * 1))
#         list_tmp = []
#         for j in range(i):
#             list_tmp.append(np.linalg.norm(anchor[j] - anchor[i]))
#         if [d <= params['dis_com'] * 2 for d in list_tmp].count(True) == 0:
#             break
anchor = np.array([[5, 10], [23, 17]])

agent = np.random.uniform(0, params['scn_len'], (params['agent_num'], 2 * 1))
# num_ = 0
for i in range(0, params['agent_num']):
    while True:
        agent[i] = np.random.randint(0, params['scn_len'], (1, 2 * 1))

        if (agent[i, 0] - anchor[0, 0]) ** 2 + (agent[i, 1] - anchor[0, 1]) ** 2 <= params['dis_com'] ** 2 and \
                (agent[i, 0] - anchor[1, 0]) ** 2 + (agent[i, 1] - anchor[1, 1]) ** 2 <= params['dis_com'] ** 2:
            break
# for i in range(0, params['anchor_num']):
#     while True:
#         anchor[i] = np.random.uniform(0, params['scn_len'], (1, 2 * 1))
#         list_tmp = []
#         for j in range(i):
#             list_tmp.append(np.linalg.norm(anchor[j] - anchor[i]))
#         if [d <= params['station_dis_min'] for d in list_tmp].count(True) == 0:
#             break
# a = anchor[1]
# num = 0
# for i in range(20):
#     anchor1_dis[i] = np.linalg.norm(anchor[0] - anchor[i])
#     anchor2_dis[i] = np.linalg.norm(anchor[1] - anchor[i])
# for i in range(anchor1_dis.shape[0]):
#     if anchor1_dis[i] <= 40 and anchor1_dis[i] <= 40:
#         num += 1
#
# num_ = 0
# for i in range(-10, 21):
#     for j in range(-10, 21):
#         if i ** 2 + j ** 2 <= 400:
#             num_ += 1

# dot7 = plot(agent[:, 0], agent[:, 1], 'r*')
# dot8 = plot(anchor[2:, 0], anchor[2:, 1], 'bo')
# dot9 = plot(anchor[:2, 0], anchor[:2, 1], 'k^')
# plt.show()
# agent = np.array([[16, 21], [22, 27], [40, 33]])

UE1_set = np.zeros((0, 2))
UE2_set = np.zeros((0, 2))
UE3_set = np.zeros((0, 2))
UE_num = 3
# signal = 0
sum_err_check_min = 100000
var_sum = 10000000
for UE1_x in range(anchor[0, 0] - params['dis_com'], anchor[0, 0] + params['dis_com'], params['grid_size']):
    for UE1_y in range(anchor[0, 1] - params['dis_com'], anchor[0, 1] + params['dis_com'], params['grid_size']):
        if (UE1_x - anchor[0, 0]) ** 2 + (UE1_y - anchor[0, 1]) ** 2 <= params['dis_com'] ** 2 and 0 <= UE1_x <= params['scn_len'] and 0 <= UE1_y <= params['scn_len']:

            UE_est = np.array([UE1_x, UE1_y])
            UE1_set = np.vstack((UE1_set, UE_est))
            UE2_set = np.vstack((UE2_set, UE_est))
            UE3_set = np.vstack((UE3_set, UE_est))
# num_num = 0
list_error_tmp = []
list_var_tmp = []

bs_tmp1 = np.zeros((0, 2))
bs_tmp2 = np.zeros((0, 2))
bs_tmp3 = np.zeros((0, 2))

ue_tmp1 = np.zeros((0, 6))
ue_tmp2 = np.zeros((0, 6))
ue_tmp3 = np.zeros((0, 6))

for i in tqdm(range(UE1_set.shape[0])):
    for j in range(UE2_set.shape[0]):
        for k in range(UE3_set.shape[0]):

            UE1_est = UE1_set[i]
            UE2_est = UE2_set[j]
            UE3_est = UE3_set[k]
            # UE1_est = agent[0]
            # UE2_est = agent[1]
            # UE3_est = agent[2]

            R1_2 = np.linalg.norm(anchor[0] - UE1_est) \
                   - (np.linalg.norm(anchor[0] - agent[0]) - np.linalg.norm(anchor[1] - agent[0]))
            R2_2 = np.linalg.norm(anchor[0] - UE2_est) \
                   - (np.linalg.norm(anchor[0] - agent[1]) - np.linalg.norm(anchor[1] - agent[1]))
            R3_2 = np.linalg.norm(anchor[0] - UE3_est) \
                   - (np.linalg.norm(anchor[0] - agent[2]) - np.linalg.norm(anchor[1] - agent[2]))

            R_2 = np.array([[R1_2], [R2_2], [R3_2]])
            UE_est_123 = np.array([UE1_est, UE2_est, UE3_est])

            # 利用UE两两相交计算出BS可能的位置，并记录在矩阵BS2_arr中
            index_2_arr = [0, 0, 0]
            BS2_list = np.linspace(100, 100 * UE_num * (UE_num - 1) / 2 * 4, int(UE_num * (UE_num - 1) / 2 * 4))
            BS2_arr = np.reshape(BS2_list, (int(UE_num * (UE_num - 1) / 2), 4))
            n2 = -1
            for a in range(UE_num):
                for b in range(a+1, UE_num):
                    n2 = n2 + 1
                    dis_ab = np.linalg.norm(UE_est_123[a] - UE_est_123[b])
                    if abs(R_2[a] - R_2[b]) <= dis_ab <= R_2[a] + R_2[b]:
                        BS3_est_x_temp = (((R_2[a] ** 2 - R_2[b] ** 2 + dis_ab ** 2) / (2 * dis_ab)) / dis_ab) * \
                                         (UE_est_123[b, 0] - UE_est_123[a, 0]) + UE_est_123[a, 0]
                        BS3_est_y_temp = (((R_2[a] ** 2 - R_2[b] ** 2 + dis_ab ** 2) / (2 * dis_ab)) / dis_ab) * \
                                         (UE_est_123[b, 1] - UE_est_123[a, 1]) + UE_est_123[a, 1]

                        x1 = BS3_est_x_temp - ((R_2[a] ** 2 - ((R_2[a] ** 2 - R_2[b] ** 2 + dis_ab ** 2) / (2 * dis_ab)) ** 2) / (1 + ((UE_est_123[b, 0] - UE_est_123[a, 0]) ** 2 / (UE_est_123[b, 1] - UE_est_123[a, 1]) ** 2))) ** (1 / 2)
                        y1 = BS3_est_y_temp - ((UE_est_123[b, 0] - UE_est_123[a, 0]) / (UE_est_123[b, 1] - UE_est_123[a, 1])) * (x1 - BS3_est_x_temp)
                        x2 = BS3_est_x_temp + ((R_2[a] ** 2 - ((R_2[a] ** 2 - R_2[b] ** 2 + dis_ab ** 2) / (2 * dis_ab)) ** 2) / (1 + ((UE_est_123[b, 0] - UE_est_123[a, 0]) ** 2 / (UE_est_123[b, 1] - UE_est_123[a, 1]) ** 2))) ** (1 / 2)
                        y2 = BS3_est_y_temp - ((UE_est_123[b, 0] - UE_est_123[a, 0]) / (UE_est_123[b, 1] - UE_est_123[a, 1])) * (x2 - BS3_est_x_temp)

                        BS2_arr[n2, 0] = x1
                        BS2_arr[n2, 1] = y1
                        BS2_arr[n2, 2] = x2
                        BS2_arr[n2, 3] = y2

                        if R_2[a] + R_2[b] < dis_ab:
                            BS3_est_x_temp = ((dis_ab + R_2[a] - R_2[b]) / (2 * dis_ab)) * (UE_est_123[b, 0] - UE_est_123[a, 0]) + UE_est_123[a, 0]
                            BS3_est_y_temp = ((dis_ab + R_2[a] - R_2[b]) / (2 * dis_ab)) * (UE_est_123[b, 1] - UE_est_123[a, 1]) + UE_est_123[a, 1]

                            BS2_arr[n2, 0] = BS3_est_x_temp
                            BS2_arr[n2, 1] = BS3_est_y_temp

                        if dis_ab < abs(R_2[a] - R_2[b]):

                            BS3_est_x_temp = ((R_2[a] + R_2[b] + dis_ab) / (2 * dis_ab)) * (UE_est_123[b, 0] - UE_est_123[a, 0]) + UE_est_123[a, 0]
                            BS3_est_y_temp = ((R_2[a] + R_2[b] + dis_ab) / (2 * dis_ab)) * (UE_est_123[b, 1] - UE_est_123[a, 1]) + UE_est_123[a, 1]

                            BS2_arr[n2, 0] = BS3_est_x_temp
                            BS2_arr[n2, 1] = BS3_est_y_temp
            # BS2_arr中BS位置的聚集程度
            var2_min = 1000000
            for p1 in range(2):
                if -3 < BS2_arr[0, p1 * 2] < params['scn_len'] + 3 and -3 < BS2_arr[0, p1 * 2 + 1] < params['scn_len'] + 3:
                    for p2 in range(2):
                        if -3 < BS2_arr[1, p2 * 2] < params['scn_len'] + 3 and -3 < BS2_arr[1, p2 * 2 + 1] < params['scn_len'] + 3:
                            for p3 in range(2):
                                if -3 < BS2_arr[2, p3 * 2] < params['scn_len'] + 3 and -3 < BS2_arr[2, p3 * 2 + 1] < params['scn_len'] + 3:

                                    var2 = np.var([BS2_arr[0, p1 * 2], BS2_arr[1, p2 * 2], BS2_arr[2, p3 * 2]]) + \
                                                np.var([BS2_arr[0, p1 * 2 + 1], BS2_arr[1, p2 * 2 + 1], BS2_arr[2, p3 * 2 + 1]])
                                    index2_arr = [p1, p2, p3]
                                    if var2 < var2_min:
                                        var2_min = var2
                                        index_2_arr = index2_arr

            BS2_sum = [0, 0]
            for ii in range(BS2_arr.shape[0]):
                BS2_sum = BS2_sum + BS2_arr[ii, 2 * index_2_arr[ii]: 2 * index_2_arr[ii] + 2]

            BS2_final = BS2_sum / BS2_arr.shape[0]

            # UE_true_arr = np.array([agent[0], agent[1], agent[2]])
            UE_est_arr = np.array([UE1_est, UE2_est, UE3_est])
            # BS_true_arr = np.array([anchor[0], anchor[1]])
            BS_est_arr = np.array([anchor[0], BS2_final])
            sum_err_check = 0
            for b1 in range(1):
                for b2 in range(b1 + 1, 2):
                    for u1 in range(0, 3):
                        sum_err_check = sum_err_check + abs((np.linalg.norm(anchor[b1, :] - agent[u1, :])-np.linalg.norm(anchor[b2, :] - agent[u1, :])) -
                                                        (np.linalg.norm(BS_est_arr[b1, :] - UE_est_arr[u1, :]) - np.linalg.norm(BS_est_arr[b2, :] - UE_est_arr[u1, :])))

            if sum_err_check < sum_err_check_min:
                sum_err_check_min = sum_err_check
                error_BS_check = abs(np.linalg.norm(BS2_final - anchor[1]))
                error_UE_check = (abs(np.linalg.norm(UE1_est - agent[0])) + abs(np.linalg.norm(UE2_est - agent[1])) + abs(np.linalg.norm(UE3_est - agent[2]))) / 3
                bs_arr_check = np.array([BS2_final])
                ue_arr_check = UE_est_arr

        # % dis_12 = get_distace(BS1_true, UE2_est) - ...
        # % (get_distace(BS1_true, UE2_true) - get_distace(BS2_final, UE2_true));
        #     sum_err_check = 0
            if var2_min + sum_err_check_min < 0.1:
                list_var_tmp.append(var2_min + sum_err_check_min)
                list_error_tmp.append(abs(np.linalg.norm(BS2_final - anchor[1])))
                bs_tmp1 = np.vstack((bs_tmp1, BS2_final))
                # bb = np.array([UE1_est, UE2_est, UE3_est])
                # aa = np.reshape(np.array([UE1_est, UE2_est, UE3_est]), (1, 6))
                # a = 1
                ue_tmp1 = np.vstack((ue_tmp1, np.reshape(np.array([UE1_est, UE2_est, UE3_est]), (1, 6))))

            if var2_min < var_sum:
                var_sum = var2_min
                index_2_arr_final = index_2_arr
                BS2_best = BS2_arr
                BS_2_est = BS2_final
                UE1_best3 = UE1_est
                UE2_best3 = UE2_est
                UE3_best3 = UE3_est

anchor = np.array([[40, 42], [23, 17]])

# agent = np.array([[48, 44], [30, 26], [36, 32]])
agent = np.random.uniform(0, params['scn_len'], (params['agent_num'], 2 * 1))
# num_ = 0
for i in range(0, params['agent_num']):
    while True:
        agent[i] = np.random.randint(0, params['scn_len'], (1, 2 * 1))

        if (agent[i, 0] - anchor[0, 0]) ** 2 + (agent[i, 1] - anchor[0, 1]) ** 2 <= params['dis_com'] ** 2 and \
                (agent[i, 0] - anchor[1, 0]) ** 2 + (agent[i, 1] - anchor[1, 1]) ** 2 <= params['dis_com'] ** 2:
            break
UE1_set = np.zeros((0, 2))
UE2_set = np.zeros((0, 2))
UE3_set = np.zeros((0, 2))
sum_err_check = 100000
for UE1_x in range(anchor[0, 0] - params['dis_com'], anchor[0, 0] + params['dis_com'], params['grid_size']):
    for UE1_y in range(anchor[0, 1] - params['dis_com'], anchor[0, 1] + params['dis_com'], params['grid_size']):
        if (UE1_x - anchor[0, 0]) ** 2 + (UE1_y - anchor[0, 1]) ** 2 <= params['dis_com'] ** 2 and 0 <= UE1_x <= params['scn_len'] and 0 <= UE1_y <= params['scn_len']:

            UE_est = np.array([UE1_x, UE1_y])
            UE1_set = np.vstack((UE1_set, UE_est))
            UE2_set = np.vstack((UE2_set, UE_est))
            UE3_set = np.vstack((UE3_set, UE_est))
for i in tqdm(range(UE1_set.shape[0])):
    for j in range(UE2_set.shape[0]):
        for k in range(UE3_set.shape[0]):

            UE1_est = UE1_set[i]
            UE2_est = UE2_set[j]
            UE3_est = UE3_set[k]
            # UE1_est = agent[0]
            # UE2_est = agent[1]
            # UE3_est = agent[2]

            R1_2 = np.linalg.norm(anchor[0] - UE1_est) \
                   - (np.linalg.norm(anchor[0] - agent[0]) - np.linalg.norm(anchor[1] - agent[0]))
            R2_2 = np.linalg.norm(anchor[0] - UE2_est) \
                   - (np.linalg.norm(anchor[0] - agent[1]) - np.linalg.norm(anchor[1] - agent[1]))
            R3_2 = np.linalg.norm(anchor[0] - UE3_est) \
                   - (np.linalg.norm(anchor[0] - agent[2]) - np.linalg.norm(anchor[1] - agent[2]))

            R_2 = np.array([[R1_2], [R2_2], [R3_2]])
            UE_est_123 = np.array([UE1_est, UE2_est, UE3_est])

            # 利用UE两两相交计算出BS可能的位置，并记录在矩阵BS2_arr中
            index_2_arr = [0, 0, 0]
            BS2_list = np.linspace(100, 100 * UE_num * (UE_num - 1) / 2 * 4, int(UE_num * (UE_num - 1) / 2 * 4))
            BS2_arr = np.reshape(BS2_list, (int(UE_num * (UE_num - 1) / 2), 4))
            n2 = -1
            for a in range(UE_num):
                for b in range(a+1, UE_num):
                    n2 = n2 + 1
                    dis_ab = np.linalg.norm(UE_est_123[a] - UE_est_123[b])
                    if abs(R_2[a] - R_2[b]) <= dis_ab <= R_2[a] + R_2[b]:
                        BS3_est_x_temp = (((R_2[a] ** 2 - R_2[b] ** 2 + dis_ab ** 2) / (2 * dis_ab)) / dis_ab) * \
                                         (UE_est_123[b, 0] - UE_est_123[a, 0]) + UE_est_123[a, 0]
                        BS3_est_y_temp = (((R_2[a] ** 2 - R_2[b] ** 2 + dis_ab ** 2) / (2 * dis_ab)) / dis_ab) * \
                                         (UE_est_123[b, 1] - UE_est_123[a, 1]) + UE_est_123[a, 1]

                        x1 = BS3_est_x_temp - ((R_2[a] ** 2 - ((R_2[a] ** 2 - R_2[b] ** 2 + dis_ab ** 2) / (2 * dis_ab)) ** 2) / (1 + ((UE_est_123[b, 0] - UE_est_123[a, 0]) ** 2 / (UE_est_123[b, 1] - UE_est_123[a, 1]) ** 2))) ** (1 / 2)
                        y1 = BS3_est_y_temp - ((UE_est_123[b, 0] - UE_est_123[a, 0]) / (UE_est_123[b, 1] - UE_est_123[a, 1])) * (x1 - BS3_est_x_temp)
                        x2 = BS3_est_x_temp + ((R_2[a] ** 2 - ((R_2[a] ** 2 - R_2[b] ** 2 + dis_ab ** 2) / (2 * dis_ab)) ** 2) / (1 + ((UE_est_123[b, 0] - UE_est_123[a, 0]) ** 2 / (UE_est_123[b, 1] - UE_est_123[a, 1]) ** 2))) ** (1 / 2)
                        y2 = BS3_est_y_temp - ((UE_est_123[b, 0] - UE_est_123[a, 0]) / (UE_est_123[b, 1] - UE_est_123[a, 1])) * (x2 - BS3_est_x_temp)

                        BS2_arr[n2, 0] = x1
                        BS2_arr[n2, 1] = y1
                        BS2_arr[n2, 2] = x2
                        BS2_arr[n2, 3] = y2

                        if R_2[a] + R_2[b] < dis_ab:
                            BS3_est_x_temp = ((dis_ab + R_2[a] - R_2[b]) / (2 * dis_ab)) * (UE_est_123[b, 0] - UE_est_123[a, 0]) + UE_est_123[a, 0]
                            BS3_est_y_temp = ((dis_ab + R_2[a] - R_2[b]) / (2 * dis_ab)) * (UE_est_123[b, 1] - UE_est_123[a, 1]) + UE_est_123[a, 1]

                            BS2_arr[n2, 0] = BS3_est_x_temp
                            BS2_arr[n2, 1] = BS3_est_y_temp

                        if dis_ab < abs(R_2[a] - R_2[b]):

                            BS3_est_x_temp = ((R_2[a] + R_2[b] + dis_ab) / (2 * dis_ab)) * (UE_est_123[b, 0] - UE_est_123[a, 0]) + UE_est_123[a, 0]
                            BS3_est_y_temp = ((R_2[a] + R_2[b] + dis_ab) / (2 * dis_ab)) * (UE_est_123[b, 1] - UE_est_123[a, 1]) + UE_est_123[a, 1]

                            BS2_arr[n2, 0] = BS3_est_x_temp
                            BS2_arr[n2, 1] = BS3_est_y_temp
            # BS2_arr中BS位置的聚集程度
            var2_min = 1000000
            for p1 in range(2):
                if -3 < BS2_arr[0, p1 * 2] < params['scn_len'] + 3 and -3 < BS2_arr[0, p1 * 2 + 1] < params['scn_len'] + 3:
                    for p2 in range(2):
                        if -3 < BS2_arr[1, p2 * 2] < params['scn_len'] + 3 and -3 < BS2_arr[1, p2 * 2 + 1] < params['scn_len'] + 3:
                            for p3 in range(2):
                                if -3 < BS2_arr[2, p3 * 2] < params['scn_len'] + 3 and -3 < BS2_arr[2, p3 * 2 + 1] < params['scn_len'] + 3:

                                    var2 = np.var([BS2_arr[0, p1 * 2], BS2_arr[1, p2 * 2], BS2_arr[2, p3 * 2]]) + \
                                                np.var([BS2_arr[0, p1 * 2 + 1], BS2_arr[1, p2 * 2 + 1], BS2_arr[2, p3 * 2 + 1]])
                                    index2_arr = [p1, p2, p3]
                                    if var2 < var2_min:
                                        var2_min = var2
                                        index_2_arr = index2_arr

            BS2_sum = [0, 0]
            for ii in range(BS2_arr.shape[0]):
                BS2_sum = BS2_sum + BS2_arr[ii, 2 * index_2_arr[ii]: 2 * index_2_arr[ii] + 2]

            BS2_final = BS2_sum / BS2_arr.shape[0]

            # UE_true_arr = np.array([agent[0], agent[1], agent[2]])
            UE_est_arr = np.array([UE1_est, UE2_est, UE3_est])
            # BS_true_arr = np.array([anchor[0], anchor[1]])
            BS_est_arr = np.array([anchor[0], BS2_final])
            sum_err_check = 0
            for b1 in range(1):
                for b2 in range(b1 + 1, 2):
                    for u1 in range(0, 3):
                        sum_err_check = sum_err_check + abs((np.linalg.norm(
                            anchor[b1, :] - agent[u1, :]) - np.linalg.norm(anchor[b2, :] - agent[u1, :])) -
                                                            (np.linalg.norm(
                                                                BS_est_arr[b1, :] - UE_est_arr[u1, :]) - np.linalg.norm(
                                                                BS_est_arr[b2, :] - UE_est_arr[u1, :])))

            if sum_err_check < sum_err_check_min:
                sum_err_check_min = sum_err_check
                error_BS_check = abs(np.linalg.norm(BS2_final - anchor[1]))
                error_UE_check = (abs(np.linalg.norm(UE1_est - agent[0])) + abs(
                    np.linalg.norm(UE2_est - agent[1])) + abs(np.linalg.norm(UE3_est - agent[2]))) / 3
                bs_arr_check = np.array([BS2_final])
                ue_arr_check = UE_est_arr

            if var2_min + sum_err_check_min < 0.1:
                list_var_tmp.append(var2_min + sum_err_check_min)
                list_error_tmp.append(abs(np.linalg.norm(BS2_final - anchor[1])))
                bs_tmp2 = np.vstack((bs_tmp2, BS2_final))
                ue_tmp2 = np.vstack((ue_tmp2, np.reshape(np.array([UE1_est, UE2_est, UE3_est]), (1, 6))))


var_bs_min = 100000
for i in tqdm(range(bs_tmp1.shape[0])):
    for j in range(bs_tmp2.shape[0]):
        # for k in range(bs_tmp3.shape[0]):
        var_bs = np.var([bs_tmp1[i, 0], bs_tmp2[j, 0]]) + np.var([bs_tmp1[i, 1], bs_tmp2[j, 1]])
        if var_bs < var_bs_min:
            var_bs_min = var_bs
            BS_12_est = (bs_tmp1[i] + bs_tmp2[j]) / 2
            index1 = i
            index2 = j

error_BS12 = abs(np.linalg.norm(BS_12_est - anchor[1]))


error_BS = abs(np.linalg.norm(BS_2_est - anchor[1]))
error_UE = (abs(np.linalg.norm(UE1_best3 - agent[0])) + abs(np.linalg.norm(UE2_best3 - agent[1])) + abs(np.linalg.norm(UE3_best3 - agent[2]))) / 3
arr_err_tmp = np.array(list_error_tmp)
arr_err_tmp = arr_err_tmp.reshape((arr_err_tmp.shape[0], 1))
arr_var_tmp = np.array(list_var_tmp)
arr_var_tmp = arr_var_tmp.reshape((arr_var_tmp.shape[0], 1))
a = 1
