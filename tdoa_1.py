import numpy as np
from cmaes import CMA
from cmaes import SepCMA
import math
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from numpy import ndarray
import random

def err_toda(BS,UE,x,UE_BS_dic,BS_known,BS_known_index,BS_unknown_index,UE_index):
    opt_BS = np.zeros((len(BS),2))
    opt_UE = np.zeros((len(UE),2))
    true_tdoa = np.zeros((len(UE),6))
    est_tdoa = np.zeros((len(UE),6))
    for i in range(len(BS_known_index)):
        opt_BS[int(BS_known_index[i])] = BS_known[i]
    for i in range(len(BS_unknown_index)):
        opt_BS[int(BS_unknown_index[i])][0] = x[2*i]
        opt_BS[int(BS_unknown_index[i])][1] = x[2*i+1]
    UE_loc = x[2*len(BS_unknown_index):]
    for i in range(len(UE_index)):
        opt_UE[int(UE_index[i])][0] = UE_loc[2*i]
        opt_UE[int(UE_index[i])][1] = UE_loc[2*i+1]

    #计算true_tdoa
    punish3 = 0
    for key in list(UE_BS_dic.keys()):
        index_ue = int(key)
        index_bs = UE_BS_dic[key]
        num_toa = len(index_bs)
        if num_toa == 2:
            true_tdoa[index_ue][0] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[0])])  - np.linalg.norm(UE[index_ue] - BS[int(index_bs[1])])
            est_tdoa[index_ue][0] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[0])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[1])])
        if num_toa == 3:
            true_tdoa[index_ue][0] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[0])]) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[1])])
            true_tdoa[index_ue][1] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[0])]) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[2])])
            true_tdoa[index_ue][2] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[1])]) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[2])])
            est_tdoa[index_ue][0] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[0])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[1])])
            est_tdoa[index_ue][1] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[0])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[2])])
            est_tdoa[index_ue][2] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[1])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[2])])
        if num_toa == 4:
            a = -1
            for i in range(3):
                for j in range(i+1,4):
                    a = a+1
                    true_tdoa[index_ue][a] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[i])]) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[j])])
                    est_tdoa[index_ue][a] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[i])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[j])])
        for i in range(len(index_bs)):
            if np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[i])])>20:
                punish3 += 1000
    error = np.sum(abs(true_tdoa - est_tdoa))
    #error = abs(logloss(true_tdoa,est_tdoa))
    punish1 = 0
    for i in range(len(index_bs)-1):
        for j in range(i+1,len(index_bs)):
            if np.linalg.norm(opt_BS[int(index_bs[i])]-opt_BS[int(index_bs[j])])<10:
                punish1 += 1000
    punish2 = 0
    for i in range(len(index_bs) - 1):
        for j in range(i + 1, len(index_bs)):
            if np.linalg.norm(opt_BS[int(index_bs[i])]-opt_BS[int(index_bs[j])])>20:
                punish2 += 1000

    cost = error + punish1 + punish2 + punish3
    return cost

def three_infer_one(BS_update,BS,UE,BS_index,index_an_ag,BS_located,min):

    UE_BS_list = []
    UE_BS_dic = {}

    # 已知BS与UE的信息
    BS_known = np.array([BS_update[int(BS_index[0])], BS_update[int(BS_index[1])], BS_update[int(BS_index[2])]])
    BS_known_index = np.array([BS_index[0], BS_index[1], BS_index[2]])
    BS_unknown_index = np.array(BS_index[3:])
    UE_index = []

    # index排序
    BS_index = np.sort(BS_index)
    index_an_ag = np.sort(index_an_ag, axis=1)
    BS_unknown_index = np.sort(BS_unknown_index)
    BS_unknown = []
    UE_known = []
    for i in range(len(BS_unknown_index)):
        BS_unknown.append(BS[int(BS_unknown_index[i])])

    # 用字典存储所有和场景内BS有关联的UE
    for i in range(len(index_an_ag)):
        for j in range(4):
            if index_an_ag[i][j] in BS_index:
                UE_BS_list.append([i, index_an_ag[i][j]])
                UE_BS_dic.setdefault(i, []).append(index_an_ag[i][j])

    # 删除信息不足或信息重复UE的键值对
    UE_BS_pair = []
    count = 0
    key_list = list(UE_BS_dic.keys())
    random.shuffle(key_list)
    new_dic = {}
    for key in key_list:
        new_dic[key] = UE_BS_dic.get(key)
    for key in list(new_dic.keys()):
        if len(new_dic[key]) < 4 :
            del new_dic[key]
        # elif UE_BS_dic[key] in UE_BS_pair:
        #    del UE_BS_dic[key]
        #   UE_BS_pair.append(UE_BS_dic[key])
    for key in list(new_dic.keys()):
        count = count + 1
        if count>4:
            del new_dic[key]
        else:
            UE_index.append(int(key))
    for i in range(len(UE_index)):
        UE_known.append(UE[int(UE_index[i])].tolist())
    #UE_known = np.array(UE_known)
    num = 0
    while (1):
        best_cost = 9999
        gen = 300
        n_ue = len(new_dic)
        n_dim = 2 * (len(BS_index) + n_ue) - 6
        lower_bounds = 0
        upper_bounds = 50
        bounds = np.zeros((n_dim, 2))
        for i in range(n_dim):
            bounds[i, :] = np.array([lower_bounds, upper_bounds])
        sigma = 0.3 * (upper_bounds - lower_bounds)
        popu = (4 + math.floor(3 * math.log(n_dim)))
        mean = lower_bounds + (np.random.rand(n_dim) * (upper_bounds - lower_bounds))
        optimizer = SepCMA(mean=mean, sigma=sigma, bounds=bounds, population_size=popu)
        best_sol = np.zeros(n_dim)
        # 初始化cmaes参数
        for generation in range(gen):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = err_toda(BS, UE, x, new_dic, BS_known, BS_known_index, BS_unknown_index, UE_index)
                if (value%1000) < best_cost:
                    best_cost = value
                    best_sol = x
                solutions.append((x, value))
                print(f"#{generation} {value} ")
            optimizer.tell(solutions)
            if optimizer.should_stop():
                break
            cost.append(best_cost)
        print(best_cost)
        num = num + 1
        if (best_cost%1000) < min :
            num = 0
            break
        elif num>20:
            num = 0
            break
    BS_located.append(int(BS_unknown_index[0]))
    return best_sol,best_cost,BS_unknown,BS_known,UE_known,BS_located

def two_infer_two(BS_update,BS,UE,BS_index,index_an_ag,BS_located,min):

    UE_BS_list = []
    UE_BS_dic = {}

    # 已知BS与UE的信息
    BS_known = np.array([BS_update[int(BS_index[0])], BS_update[int(BS_index[1])]])
    BS_known_index = np.array([BS_index[0], BS_index[1]])
    BS_unknown_index = np.array(BS_index[2:])
    UE_index = []

    # index排序
    BS_index = np.sort(BS_index)
    index_an_ag = np.sort(index_an_ag, axis=1)
    BS_unknown_index = np.sort(BS_unknown_index)
    BS_unknown = []
    UE_known = []
    for i in range(len(BS_unknown_index)):
        BS_unknown.append(BS[int(BS_unknown_index[i])])

    # 用字典存储所有和场景内BS有关联的UE
    for i in range(len(index_an_ag)):
        for j in range(4):
            if index_an_ag[i][j] in BS_index:
                UE_BS_list.append([i, index_an_ag[i][j]])
                UE_BS_dic.setdefault(i, []).append(index_an_ag[i][j])

    # 删除信息不足或信息重复UE的键值对
    UE_BS_pair = []
    count = 0
    key_list = list(UE_BS_dic.keys())
    random.shuffle(key_list)
    new_dic = {}
    for key in key_list:
        new_dic[key] = UE_BS_dic.get(key)
    for key in list(new_dic.keys()):
        if len(new_dic[key]) < 4 :
            del new_dic[key]
        # elif UE_BS_dic[key] in UE_BS_pair:
        #    del UE_BS_dic[key]
        #   UE_BS_pair.append(UE_BS_dic[key])
    for key in list(new_dic.keys()):
        count = count + 1
        if count>4:
            del new_dic[key]
        else:
            UE_index.append(int(key))
    for i in range(len(UE_index)):
        UE_known.append(UE[int(UE_index[i])].tolist())
    #UE_known = np.array(UE_known)
    num = 0
    while (1):
        # 初始化cmaes参数
        gen = 300
        n_ue = len(new_dic)
        n_dim = 2 * (len(BS_index) + n_ue) - 4
        best_cost = 9999
        lower_bounds = 0
        upper_bounds = 50
        best_sol = np.zeros(n_dim)
        bounds = np.zeros((n_dim, 2))
        for i in range(n_dim):
            bounds[i, :] = np.array([lower_bounds, upper_bounds])
        sigma = 0.3 * (upper_bounds - lower_bounds)
        popu = (4 + math.floor(3 * math.log(n_dim)))
        mean = lower_bounds + (np.random.rand(n_dim) * (upper_bounds - lower_bounds))
        optimizer = SepCMA(mean=mean, sigma=sigma, bounds=bounds, population_size=popu)
        for generation in range(gen):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = err_toda(BS, UE, x, new_dic, BS_known, BS_known_index, BS_unknown_index, UE_index)
                if value < best_cost:
                    best_cost = value
                    best_sol = x
                solutions.append((x, value))
                print(f"#{generation} {value} ")
            optimizer.tell(solutions)
            if optimizer.should_stop():
                break
            cost.append(best_cost)
        print(best_cost)
        if (best_cost%1000) < min :
            num = 0
            break
        elif num>50:
            num = 0
            break
        num = num+1
    BS_located.append(int(BS_unknown_index[0]))
    BS_located.append(int(BS_unknown_index[1]))
    return best_sol,best_cost,BS_unknown,BS_known,UE_known,BS_located

if __name__ == "__main__":
    strat = time.time()
    rmse = np.zeros(6)
    mse = np.zeros((6,5))
    for turn in range(5):
        _rmse = []
        cost = []
        BS_located = []
        BS_unknown_index = []
        BS_unknown_location = []
        BS_unknown_true = []
        index_an_ag_dic = {}
        new_UE_BS = []
        old_UE_BS = []
        new_BS = []
        UE_location_true = []
        UE_pre = []
        BS_update = np.zeros((20,2))
        #数据初始化
        BS = np.loadtxt('tdoa_data\\1\\anchor.txt',delimiter=',')
        UE = np.loadtxt('tdoa_data\\1\\agent.txt',delimiter=',')
        BS_index = np.loadtxt('tdoa_data\\1\\index.txt',delimiter=',')
        index_an_ag = np.loadtxt('tdoa_data\\1\\index_an_ag4.txt',delimiter=',')

        for i in range(len(UE)):
            for j in range(4):
                index_an_ag_dic.setdefault(i, []).append(index_an_ag[i][j])
        BS_known_index = np.array([BS_index[0], BS_index[1], BS_index[2]])#BS_known_index = np.array([BS_index[0], BS_index[1], BS_index[2]])####BS_known_index = np.array([BS_index[0], BS_index[1], BS_index[2]])
        for i in range(len(BS_known_index)):
            BS_located.append(int(BS_known_index[i]))
            BS_update[int(BS_known_index[i])] = BS[int(BS_known_index[i])]
        #BS_unknown_index.append(int(BS_index[2]))#
        BS_unknown_index.append(int(BS_index[3]))
        min = 0.1
        best_sol,best_cost,BS_unknown,BS_known,UE_known,BS_located = three_infer_one(BS_update,BS,UE,BS_index,index_an_ag,BS_located,min)
        for i in range(len(UE_known)):
            UE_location_true.append(UE_known[i])
        ue_pre = best_sol[2:]#ue_pre = best_sol[4:]###ue_pre = best_sol[2:]
        for i in range(4):
            UE_pre.append([ue_pre[2*i],ue_pre[2 * i+1]])
        BS_known_1 = BS_known
        BS_unknown_location.append([best_sol[0],best_sol[1]])
        #BS_unknown_location.append([best_sol[2], best_sol[3]])
        #BS_unknown_true.append(BS[int(BS_index[2])])
        BS_unknown_true.append(BS[int(BS_index[3])])
        BS_update[int(BS_unknown_index[0])] = np.array([best_sol[0],best_sol[1]])
        #BS_update[int(BS_unknown_index[1])] = np.array([best_sol[2], best_sol[3]])
        #外推场景确定
        for key in index_an_ag_dic.keys():
            count = 0
            values = index_an_ag_dic[key]
            for i in range(len(values)):
                a = values[i]
                if a in BS_located:
                    count = count + 1
                elif a not in BS_located:
                    new = a
            if count == 3 and values not in new_UE_BS and new not in new_BS:
                new_UE_BS.append(values)
                new_BS.append(new)
        for i in range(len(new_UE_BS)):
            for j in range(len(new_UE_BS[i])):
                if new_UE_BS[i][j] in new_BS:
                    new_UE_BS[i].append(new_UE_BS[i][j])
                    new_UE_BS[i].remove(new_UE_BS[i][j])
                    break
        for i in range(len(new_UE_BS)):
            old_UE_BS.append(new_UE_BS[i])
        new_UE_BS = [[8.0,11.0,15.0,6.0]]
        new_BS = [6.0]
        for i in range(len(new_UE_BS)):
            BS_index = new_UE_BS[i]
            BS_index = np.array(BS_index)
            BS_unknown_index.append(int(BS_index[3:]))
            min = 4
            best_sol, best_cost, BS_unknown, BS_known, UE_known, BS_located = three_infer_one(BS_update,BS, UE, BS_index, index_an_ag,BS_located,min)
            for i in range(len(UE_known)):
                UE_location_true.append(UE_known[i])
            ue_pre = best_sol[2:]
            for i in range(4):
                UE_pre.append([ue_pre[2 * i], ue_pre[2 * i + 1]])
            BS_unknown_location.append([best_sol[0], best_sol[1]])
            BS_update[int(BS_unknown_index[-1])] = np.array([best_sol[0], best_sol[1]])
        for i in range(len(new_BS)):
            BS_unknown_true.append(BS[int(new_BS[i])])
        new_UE_BS = []
        new_BS = []
        for i in range(len(BS_unknown_true)):
            mse[0][turn] += np.linalg.norm(BS_unknown_true[i]-BS_unknown_location[i])**2
        total_num = len(BS_unknown_true)
        mse[0][turn] = math.sqrt(mse[0][turn]/total_num)
        #rmse[turn] = math.sqrt(mse/total_num)
#1
        for key in index_an_ag_dic.keys():
            count = 0
            values = index_an_ag_dic[key]
            for i in range(len(values)):
                a = values[i]
                if a in BS_located:
                    count = count + 1
                elif a not in BS_located:
                    new = a
            if count == 3 and values not in new_UE_BS and new not in new_BS and values not in old_UE_BS:
                new_UE_BS.append(values)
                new_BS.append(new)
        for i in range(len(new_UE_BS)):
            for j in range(len(new_UE_BS[i])):
                if new_UE_BS[i][j] in new_BS:
                    new_UE_BS[i].append(new_UE_BS[i][j])
                    new_UE_BS[i].remove(new_UE_BS[i][j])
                    break

        new_UE_BS = [[6.0,11.0,8.0,2.0],[8.0,10.0,6.0,5.0],[8.0,10.0,15.0,17.0]]
        new_BS = [2.0,5.0,17.0]
        for i in range(len(new_UE_BS)):
            BS_index = new_UE_BS[i]
            BS_index = np.array(BS_index)
            BS_unknown_index.append(int(BS_index[3:]))
            min = 10
            best_sol, best_cost, BS_unknown, BS_known, UE_known, BS_located = three_infer_one(BS_update, BS, UE, BS_index,
                                                                                              index_an_ag, BS_located, min)
            for i in range(len(UE_known)):
                UE_location_true.append(UE_known[i])
            ue_pre = best_sol[2:]
            for i in range(4):
                UE_pre.append([ue_pre[2 * i], ue_pre[2 * i + 1]])
            BS_unknown_location.append([best_sol[0], best_sol[1]])
            BS_update[int(BS_unknown_index[-1])] = np.array([best_sol[0], best_sol[1]])
        for i in range(len(new_BS)):
            BS_unknown_true.append(BS[int(new_BS[i])])
        new_UE_BS = []
        new_BS = []
        for i in range(len(BS_unknown_true)):
            mse[1][turn] += np.linalg.norm(BS_unknown_true[i]-BS_unknown_location[i])**2
        total_num = len(BS_unknown_true)
        mse[1][turn] = math.sqrt(mse[1][turn]/total_num)
#2
        for key in index_an_ag_dic.keys():
            count = 0
            values = index_an_ag_dic[key]
            for i in range(len(values)):
                a = values[i]
                if a in BS_located:
                    count = count + 1
                elif a not in BS_located:
                    new = a
            if count == 3 and values not in new_UE_BS and new not in new_BS:
                new_UE_BS.append(values)
                new_BS.append(new)
        for i in range(len(new_UE_BS)):
            for j in range(len(new_UE_BS[i])):
                if new_UE_BS[i][j] in new_BS:
                    new_UE_BS[i].append(new_UE_BS[i][j])
                    new_UE_BS[i].remove(new_UE_BS[i][j])
                    break
        for i in range(len(new_UE_BS)):
            old_UE_BS.append(new_UE_BS[i])

        new_UE_BS = [ [2.0, 6.0, 5.0, 0.0], [17.0, 15.0, 10.0, 14.0],[10.0,5.0,8.0,9.0]]
        new_BS = [ 0.0,14.0,9.0]
        for i in range(len(new_UE_BS)):
            BS_index = new_UE_BS[i]
            BS_index = np.array(BS_index)
            BS_unknown_index.append(int(BS_index[3:]))
            min = 100
            best_sol, best_cost, BS_unknown, BS_known, UE_known, BS_located = three_infer_one(BS_update, BS, UE,
                                                                                              BS_index,
                                                                                              index_an_ag, BS_located,
                                                                                              min)
            for i in range(len(UE_known)):
                UE_location_true.append(UE_known[i])
            ue_pre = best_sol[2:]
            for i in range(4):
                UE_pre.append([ue_pre[2 * i], ue_pre[2 * i + 1]])
            BS_unknown_location.append([best_sol[0], best_sol[1]])
            BS_update[int(BS_unknown_index[-1])] = np.array([best_sol[0], best_sol[1]])
        for i in range(len(new_BS)):
            BS_unknown_true.append(BS[int(new_BS[i])])
        new_UE_BS = []
        new_BS = []
        for i in range(len(BS_unknown_true)):
            mse[2][turn] += np.linalg.norm(BS_unknown_true[i]-BS_unknown_location[i])**2
        total_num = len(BS_unknown_true)
        mse[2][turn] = math.sqrt(mse[2][turn]/total_num)

#3
        for key in index_an_ag_dic.keys():
            count = 0
            values = index_an_ag_dic[key]
            for i in range(len(values)):
                a = values[i]
                if a in BS_located:
                    count = count + 1
                elif a not in BS_located:
                    new = a
            if count == 3 and values not in new_UE_BS and new not in new_BS:
                new_UE_BS.append(values)
                new_BS.append(new)
        for i in range(len(new_UE_BS)):
            for j in range(len(new_UE_BS[i])):
                if new_UE_BS[i][j] in new_BS:
                    new_UE_BS[i].append(new_UE_BS[i][j])
                    new_UE_BS[i].remove(new_UE_BS[i][j])
                    break
        for i in range(len(new_UE_BS)):
            old_UE_BS.append(new_UE_BS[i])

        new_UE_BS = [[2.0, 0.0, 5.0, 3.0], [17.0, 14.0, 10.0, 19.0], [14.0, 9.0, 10.0, 13.0]]
        new_BS = [3.0, 19.0,13.0]
        for i in range(len(new_UE_BS)):
            BS_index = new_UE_BS[i]
            BS_index = np.array(BS_index)
            BS_unknown_index.append(int(BS_index[3:]))
            min = 150
            best_sol, best_cost, BS_unknown, BS_known, UE_known, BS_located = three_infer_one(BS_update, BS, UE,
                                                                                              BS_index,
                                                                                              index_an_ag, BS_located,
                                                                                              min)
            for i in range(len(UE_known)):
                UE_location_true.append(UE_known[i])
            ue_pre = best_sol[2:]
            for i in range(4):
                UE_pre.append([ue_pre[2 * i], ue_pre[2 * i + 1]])
            BS_unknown_location.append([best_sol[0], best_sol[1]])
            BS_update[int(BS_unknown_index[-1])] = np.array([best_sol[0], best_sol[1]])
        for i in range(len(new_BS)):
            BS_unknown_true.append(BS[int(new_BS[i])])
        new_UE_BS = []
        new_BS = []
        for i in range(len(BS_unknown_true)):
            mse[3][turn] += np.linalg.norm(BS_unknown_true[i]-BS_unknown_location[i])**2
        total_num = len(BS_unknown_true)
        mse[3][turn] = math.sqrt(mse[3][turn]/total_num)
#4
        for key in index_an_ag_dic.keys():
            count = 0
            values = index_an_ag_dic[key]
            for i in range(len(values)):
                a = values[i]
                if a in BS_located:
                    count = count + 1
                elif a not in BS_located:
                    new = a
            if count == 3 and values not in new_UE_BS and new not in new_BS:
                new_UE_BS.append(values)
                new_BS.append(new)
        for i in range(len(new_UE_BS)):
            for j in range(len(new_UE_BS[i])):
                if new_UE_BS[i][j] in new_BS:
                    new_UE_BS[i].append(new_UE_BS[i][j])
                    new_UE_BS[i].remove(new_UE_BS[i][j])
                    break
        for i in range(len(new_UE_BS)):
            old_UE_BS.append(new_UE_BS[i])

        new_UE_BS = [[19.0,17.0,14.0,18.0], [0.0, 2.0, 3.0, 1.0],[3.0,5.0,9.0,7.0]]
        new_BS = [18.0,1.0,7.0]
        for i in range(len(new_UE_BS)):
            BS_index = new_UE_BS[i]
            BS_index = np.array(BS_index)
            BS_unknown_index.append(int(BS_index[3:]))
            min = 250
            best_sol, best_cost, BS_unknown, BS_known, UE_known, BS_located = three_infer_one(BS_update, BS, UE,
                                                                                              BS_index,
                                                                                              index_an_ag, BS_located,
                                                                                              min)
            for i in range(len(UE_known)):
                UE_location_true.append(UE_known[i])
            ue_pre = best_sol[2:]
            for i in range(4):
                UE_pre.append([ue_pre[2 * i], ue_pre[2 * i + 1]])
            BS_unknown_location.append([best_sol[0], best_sol[1]])
            BS_update[int(BS_unknown_index[-1])] = np.array([best_sol[0], best_sol[1]])
        for i in range(len(new_BS)):
            BS_unknown_true.append(BS[int(new_BS[i])])
        new_UE_BS = []
        new_BS = []
        for i in range(len(BS_unknown_true)):
            mse[4][turn] += np.linalg.norm(BS_unknown_true[i]-BS_unknown_location[i])**2
        total_num = len(BS_unknown_true)
        mse[4][turn] = math.sqrt(mse[4][turn]/total_num)

#5
        for key in index_an_ag_dic.keys():
            count = 0
            values = index_an_ag_dic[key]
            for i in range(len(values)):
                a = values[i]
                if a in BS_located:
                    count = count + 1
                elif a not in BS_located:
                    new = a
            if count == 3 and values not in new_UE_BS and new not in new_BS:
                new_UE_BS.append(values)
                new_BS.append(new)
        for i in range(len(new_UE_BS)):
            for j in range(len(new_UE_BS[i])):
                if new_UE_BS[i][j] in new_BS:
                    new_UE_BS[i].append(new_UE_BS[i][j])
                    new_UE_BS[i].remove(new_UE_BS[i][j])
                    break
        for i in range(len(new_UE_BS)):
            old_UE_BS.append(new_UE_BS[i])

        new_UE_BS = [ [13.0, 18.0, 9.0, 16.0],[7.0,3.0,13.0,12.0]]
        new_BS = [16.0,12.0]
        for i in range(len(new_UE_BS)):
            BS_index = new_UE_BS[i]
            BS_index = np.array(BS_index)
            BS_unknown_index.append(int(BS_index[3:]))
            min = 300
            best_sol, best_cost, BS_unknown, BS_known, UE_known, BS_located = three_infer_one(BS_update, BS, UE,
                                                                                              BS_index,
                                                                                              index_an_ag, BS_located,
                                                                                              min)
            for i in range(len(UE_known)):
                UE_location_true.append(UE_known[i])
            ue_pre = best_sol[2:]
            for i in range(4):
                UE_pre.append([ue_pre[2 * i], ue_pre[2 * i + 1]])
            BS_unknown_location.append([best_sol[0], best_sol[1]])
            BS_update[int(BS_unknown_index[-1])] = np.array([best_sol[0], best_sol[1]])
        for i in range(len(new_BS)):
            BS_unknown_true.append(BS[int(new_BS[i])])
        new_UE_BS = []
        new_BS = []
        for i in range(len(BS_unknown_true)):
            mse[5][turn] += np.linalg.norm(BS_unknown_true[i]-BS_unknown_location[i])**2
        total_num = len(BS_unknown_true)
        mse[5][turn] = math.sqrt(mse[5][turn]/total_num)
        #ue_pre = np.array(best_sol[2:])
        #UE_pre = np.zeros((40,2))
        #for i in range(3):
        #    UE_pre[i, 0] = ue_pre[2*i]
        #    UE_pre[i, 1] = ue_pre[2 * i+1]
        #rmse = math.sqrt((np.linalg.norm(bs[0]-BS_unknown[0])**2))
        #_rmse.append(rmse)
        print('1')
    end = time.time()
    full_time = end - strat
    _rmse = np.average(mse,axis=1)
    print(mse)
    print(_rmse)
    print('number=' + str(len(BS_located)))
    print(best_sol)
    print(best_cost)
    print(full_time)
    #print(rmse)
    #plt.figure()
    #plt.ylabel('cost')
    #plt.xlabel('itr')
    #plt.plot(itr,cost)
    #plt.show()
    BS_unknown_true = np.array(BS_unknown_true)
    BS_unknown_location = np.array(BS_unknown_location)
    UE_location_true = np.array(UE_location_true)
    UE_pre = np.array(UE_pre)
    plt.figure()
    BS_unknown = np.array(BS_unknown)
    plt.axis([0, 50, 0, 50])
    dot1 = plot(BS_known_1[:,0],BS_known_1[:,1],'bo',color = 'r',label = 'BS(known)')
    dot2 = plot(BS_unknown_true[:,0],BS_unknown_true[:,1],'bo',color = 'b',label = 'BS(unknown)')
    dot3 = plot(BS_unknown_location[:,0],BS_unknown_location[:,1],'bo',color = 'y',label = 'BS(prediction)')
    dot4 = plot(UE_location_true[:, 0], UE_location_true[:, 1], '*', color='g', label='UE')
    dot5 = plot(UE_pre[:, 0], UE_pre[:, 1], '*', color='y', label='UE(prediction)')
    #dot4 = plot(UE_pre[:, 0], UE_pre[:, 1], '*', color='y', label='UE(truth)')
    plt.legend(loc="upper left")
    plt.show()



