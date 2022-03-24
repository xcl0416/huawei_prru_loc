import numpy as np


def symm(bs, BS1_true, BS2_true):       # 求bs关于BS1_true和BS2_true连线的对称解
    dis_12 = np.linalg.norm(BS1_true - BS2_true)
    bs_sym = np.zeros(bs.shape)
    for i in range(bs.shape[0]):
        R_1 = np.linalg.norm(bs[i, :] - BS1_true)
        R_2 = np.linalg.norm(bs[i, :] - BS2_true)

        x_temp = (((R_1 ** 2 - R_2 ** 2 + dis_12 ** 2) / (2 * dis_12)) / dis_12) * (BS2_true[0] - BS1_true[0]) + BS1_true[0]
        y_temp = (((R_1 ** 2 - R_2 ** 2 + dis_12 ** 2) / (2 * dis_12)) / dis_12) * (BS2_true[1] - BS1_true[1]) + BS1_true[1]

        if BS1_true[1] == BS2_true[1]:
            x1 = x_temp
            y1 = y_temp + (R_1 ** 2 - ((R_1 ** 2 - R_2 ** 2 + dis_12 ** 2) / (2 * dis_12)) ** 2) ** (1 / 2)
            x2 = x_temp
            y2 = y_temp - (R_1 ** 2 - ((R_1 ** 2 - R_2 ** 2 + dis_12 ** 2) / (2 * dis_12)) ** 2) ** (1 / 2)

        if BS1_true[1] != BS2_true[1]:
            x1 = x_temp - ((R_1 ** 2 - ((R_1 ** 2 - R_2 ** 2 + dis_12 ** 2) / (2 * dis_12)) ** 2) / (1 + ((BS2_true[0] - BS1_true[0]) ** 2 / (BS2_true[1] - BS1_true[1]) ** 2))) ** (1/2)
            y1 = y_temp - ((BS2_true[0] - BS1_true[0]) / (BS2_true[1] - BS1_true[1])) * (x1 - x_temp)
            x2 = x_temp + ((R_1 ** 2 - ((R_1 ** 2 - R_2 ** 2 + dis_12 ** 2) / (2 * dis_12)) ** 2) / (1 + ((BS2_true[0] - BS1_true[0]) ** 2 / (BS2_true[1] - BS1_true[1]) ** 2))) ** (1/2)
            y2 = y_temp - ((BS2_true[0] - BS1_true[0]) / (BS2_true[1] - BS1_true[1])) * (x2 - x_temp)

        if np.linalg.norm(bs[i, :] - [x1, y1]) > np.linalg.norm(bs[i, :] - [x2, y2]):
            bs_sym[i, :] = [x1, y1]
        if np.linalg.norm(bs[i, :] - [x1, y1]) < np.linalg.norm(bs[i, :] - [x2, y2]):
            bs_sym[i, :] = [x2, y2]

    return bs_sym

