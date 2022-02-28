import sys
import numpy as np


def avg_cal(x):
    sum_res = 0.
    cnt = 0
    for item in x:
        sum_res += item
        cnt += 1
    return sum_res / cnt


if __name__ == '__main__':
    id_list = ['01', '02', '03', '04', '05']
    id_list_cnt = 0
    # 需要求平均的数据： di spd accuracy
    avg_di_20 = []
    avg_spd_20 = []
    avg_accuracy_20 = []

    while id_list_cnt < 1:
        id_list_cnt += 1
        tmp_di = np.load('./bank-testres/di_res' + id_list[id_list_cnt] + '.npy', allow_pickle=True)
        tmp_spd = np.load('./bank-testres/spd_res' + id_list[id_list_cnt] + '.npy')
        tmp_acc = np.logical_and('./bank-testres/test_accuracy' + id_list[id_list_cnt] + '.npy')
        for item in tmp_di:
            print(tmp_di.index(item))
