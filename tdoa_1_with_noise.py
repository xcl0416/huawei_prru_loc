import numpy as np
from cmaes import CMA
from cmaes import SepCMA
import math
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from numpy import ndarray
import random

def logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()

def err_toda(BS,UE,x,UE_BS_dic,BS_known,BS_known_index,BS_unknown_index,UE_index,sig):
    mu = 0
    sig = np.sqrt(sig)
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
            true_tdoa[index_ue][0] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[0])]) + random.gauss(mu,sig) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[1])]) + random.gauss(mu,sig)
            est_tdoa[index_ue][0] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[0])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[1])])
        if num_toa == 3:
            true_tdoa[index_ue][0] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[0])]) + random.gauss(mu,sig) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[1])]) + random.gauss(mu,sig)
            true_tdoa[index_ue][1] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[0])]) + random.gauss(mu,sig) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[2])]) + random.gauss(mu,sig)
            true_tdoa[index_ue][2] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[1])]) + random.gauss(mu,sig) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[2])]) + random.gauss(mu,sig)
            est_tdoa[index_ue][0] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[0])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[1])])
            est_tdoa[index_ue][1] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[0])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[2])])
            est_tdoa[index_ue][2] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[1])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[2])])
        if num_toa == 4:
            a = -1
            for i in range(3):
                for j in range(i+1,4):
                    a = a+1
                    true_tdoa[index_ue][a] = np.linalg.norm(UE[index_ue] - BS[int(index_bs[i])]) + random.gauss(mu,sig) - np.linalg.norm(UE[index_ue] - BS[int(index_bs[j])]) + random.gauss(mu,sig)
                    est_tdoa[index_ue][a] = np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[i])]) - np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[j])])
        for i in range(len(index_bs)):
            if np.linalg.norm(opt_UE[index_ue] - opt_BS[int(index_bs[i])])>20:
                punish3 += 1000
    error = np.sum(abs(true_tdoa - est_tdoa))
    #error = abs(logloss(true_tdoa,est_tdoa))
    punish1 = 0
    for i in range(len(BS_index)-1):
        for j in range(i+1,len(BS_index)):
            if np.linalg.norm(opt_BS[int(BS_index[i])]-opt_BS[int(BS_index[j])])<10:
                punish1 += 1000
    punish2 = 0
    for i in range(len(BS_unknown_index)):
        for j in range(len(BS_known_index)):
            if np.linalg.norm(opt_BS[int(BS_unknown_index[i])]-opt_BS[int(BS_known_index[j])])>20:
                punish2 += 1000

    cost = error + punish1 + punish2 + punish3
    return cost

if __name__ == "__main__":
    noise = {}
    sig =  [0.3,0.5,1,1.6,2.5,3.9,6];
    reshold = [999,999,999,999,999,999,999]
    for k in range(len(sig)):
        _rmse = []
        for turn in range(100):
            strat = time.time()
            while(1):
                #数据初始化
                UE_BS_list = []
                UE_BS_dic = {}
                BS = np.loadtxt('tdoa_data\\1\\anchor.txt',delimiter=',')
                UE = np.loadtxt('tdoa_data\\1\\agent.txt',delimiter=',')
                BS_index: ndarray = np.loadtxt('tdoa_data\\1\\index.txt',delimiter=',')
                index_an_ag = np.loadtxt('tdoa_data\\1\\index_an_ag4.txt',delimiter=',')

                #已知BS与UE的信息
                BS_known = np.array([BS[int(BS_index[0])],BS[int(BS_index[1])]])
                BS_known_index = np.array([BS_index[0],BS_index[1]])
                BS_unknown_index = np.array(BS_index[2:])
                UE_index = []

                #index排序
                BS_index = np.sort(BS_index)
                index_an_ag = np.sort(index_an_ag,axis=1)
                BS_unknown_index = np.sort(BS_unknown_index)
                BS_unknown = []
                UE_known = []
                for i in range(len(BS_unknown_index)):
                    BS_unknown.append(BS[int(BS_unknown_index[i])])

                #用字典存储所有和场景内BS有关联的UE
                for i in range(len(index_an_ag)):
                    for j in range(4):
                        if index_an_ag[i][j] in BS_index:
                            UE_BS_list.append([i,index_an_ag[i][j]])
                            UE_BS_dic.setdefault(i,[]).append(index_an_ag[i][j])

                #删除信息不足或信息重复UE的键值对
                UE_BS_pair = []
                for key in list(UE_BS_dic.keys()):
                    if len(UE_BS_dic[key]) < 4:
                        del UE_BS_dic[key]
                    #elif UE_BS_dic[key] in UE_BS_pair:
                    #    del UE_BS_dic[key]
                    else:
                        UE_index.append(int(key))
                     #   UE_BS_pair.append(UE_BS_dic[key])
                UE_BS_dic = {23: [8.0, 10.0, 11.0, 15.0], 35: [8.0, 10.0, 11.0, 15.0], 49: [8.0, 10.0, 11.0, 15.0]}
                UE_index = [23,35,49]

                for i in range(len(UE_index)):
                    UE_known.append(UE[int(UE_index[i])])
                UE_known = np.array(UE_known)

                #初始化cmaes参数
                gen = 300
                cost = []
                itr = np.linspace(1,gen,num=gen)
                n_ue = len(UE_BS_dic)
                n_dim = 2*(len(BS_index) + n_ue) - 4
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
                        value = err_toda(BS,UE,x,UE_BS_dic,BS_known,BS_known_index,BS_unknown_index,UE_index,sig[k])
                        if value < best_cost:
                            best_cost = value
                            best_sol = x
                        solutions.append((x, value))
                        print(f"#{generation} {value} ")
                    optimizer.tell(solutions)
                    if optimizer.should_stop():
                        break
                    cost.append(best_cost)
                end = time.time()
                full_time = end-strat
                bs = np.zeros((2,2))
                bs[0] = np.array([best_sol[0],best_sol[1]])
                bs[1] = np.array([best_sol[2],best_sol[3]])
                ue_pre = np.array(best_sol[2:])
                UE_pre = np.zeros((40,2))
                for i in range(3):
                    UE_pre[i, 0] = ue_pre[2*i]
                    UE_pre[i, 1] = ue_pre[2 * i+1]
                rmse = math.sqrt((np.linalg.norm(bs[0]-BS_unknown[0])**2+np.linalg.norm(bs[1]-BS_unknown[1])**2)/2)
                if (best_cost) < (reshold[k]-5):
                    break
            _rmse.append(rmse)
        noise.setdefault(sig[k],[]).append(_rmse)
        _rmse = np.array(_rmse)
        prob = []
        a = 0
        for i in range(len(_rmse)):
            if _rmse[i]<4:
                a += 1
        prob.append(a/100)
    np.save('2_2_noise.npy', noise)
    print(prob)


    #print(best_sol)
    #print(best_cost)
    #print(full_time)
    #print(rmse)
    #print(_rmse)
    #plt.figure()
    #plt.ylabel('cost')
    '''plt.xlabel('itr')
    plt.plot(itr,cost)
    plt.show()

    plt.figure()
    BS_unknown = np.array(BS_unknown)
    plt.axis([0, 50, 0, 50])
    dot1 = plot(BS_known[:,0],BS_known[:,1],'bo',color = 'r',label = 'BS(known)')
    dot2 = plot(BS_unknown[:,0],BS_unknown[:,1],'bo',color = 'b',label = 'BS(unknown)')
    dot3 = plot(bs[:,0],bs[:,1],'bo',color = 'y',label = 'BS(prediction)')
    dot4 = plot(UE_known[:,0],UE_known[:,1],'*',color ='r',label ='UE(truth)')
    dot4 = plot(UE_pre[:, 0], UE_pre[:, 1], '*', color='y', label='UE(truth)')
    plt.legend()
    plt.show()'''