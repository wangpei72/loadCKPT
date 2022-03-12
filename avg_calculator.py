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
    # 需要求平均的数据：eoop spd accu
    avg_di_20 = []
    avg_spd_20 = []
    avg_accuracy_20 = []
    idx = 0
    avg_di = np.zeros(20, dtype=np.float32)
    avg_spd = np.zeros(20, dtype=np.float32)
    avg_eoop = np.zeros(20, dtype=np.float32)
    avg_eood = np.zeros(20, dtype=np.float32)
    # avg_acc = np.zeros(20, dtype=object)
    while id_list_cnt < 5:
        # 此处计算的是adult数据
        # tmp_di = np.load('./adult-testres-age/di_res' + id_list[id_list_cnt] + '.npy')
        # tmp_spd = np.load('./adult-testres-age/spd_res' + id_list[id_list_cnt] + '.npy')
        # tmp_eoop = np.load('./adult-testres-age/eoop_res' + id_list[id_list_cnt] + '.npy')
        # tmp_eood = np.load('./adult-testres-age/eood_res' + id_list[id_list_cnt] + '.npy')
        # tmp_acc = np.load('./adult-testres/test_accuracy' + id_list[id_list_cnt] + '.npy', allow_pickle=True)

        # 下面是计算bank的
        tmp_di = np.load('./bank-testres/di_res' + id_list[id_list_cnt] + '.npy')
        tmp_spd = np.load('./bank-testres/spd_res' + id_list[id_list_cnt] + '.npy')
        tmp_eoop = np.load('./bank-testres/eoop_res' + id_list[id_list_cnt] + '.npy')
        tmp_eood = np.load('./bank-testres/eood_res' + id_list[id_list_cnt] + '.npy')

        avg_di += tmp_di
        avg_spd += tmp_spd
        avg_eoop += tmp_eoop
        avg_eood += tmp_eood
        # avg_acc += tmp_acc
        if id_list_cnt == 4:
            avg_di /= 5.
            avg_spd /= 5.
            avg_eoop /= 5.
            avg_eood /= 5.
            # avg_acc /= 5.
            # 下面是计算adult的
            # np.save('adult-testres-age/res_avg/di_avg.npy', avg_di)
            # np.save('adult-testres-age/res_avg/spd_avg.npy', avg_spd)
            # np.save('adult-testres-age/res_avg/eoop_avg.npy', avg_eoop)
            # np.save('adult-testres-age/res_avg/eood_avg.npy', avg_eood)
            # np.save('adult-testres/res_avg/test_accu_avg.npy', avg_acc)

            # 下面是计算bank的
            np.save('bank-testres/res_avg/di_avg.npy', avg_di)
            np.save('bank-testres/res_avg/spd_avg.npy', avg_spd)
            np.save('bank-testres/res_avg/eoop_avg.npy', avg_eoop)
            np.save('bank-testres/res_avg/eood_avg.npy', avg_eood)
        id_list_cnt += 1

    print('avg calculating done. ')
# if __name__ == '__main__':
#     id_list = ['01', '02', '03', '04', '05']
#     id_list_cnt = 0
#     # 需要求平均的数据： di spd accuracy
#     avg_di_20 = []
#     avg_spd_20 = []
#     avg_accuracy_20 = []
#     idx = 0
#     avg_di = np.zeros(20, dtype=np.float32)
#     avg_spd = np.zeros(20, dtype=np.float32)
#     avg_acc = np.zeros(20, dtype=object)
#     while id_list_cnt < 5:
#         tmp_di = np.load('./bank-adult-testres/di_res' + id_list[id_list_cnt] + '.npy', allow_pickle=True)
#         tmp_spd = np.load('./bank-adult-testres/spd_res' + id_list[id_list_cnt] + '.npy')
#         # tmp_acc = np.load('./test_accuracy' + id_list[id_list_cnt] + '.npy', allow_pickle=True)
#
#         avg_di += tmp_di
#         avg_spd += tmp_spd
#         # avg_acc += tmp_acc
#         if id_list_cnt == 4:
#             avg_di /= 5.
#             avg_spd /= 5.
#             # avg_acc /= 5.
#         id_list_cnt += 1
#         np.save('bank-adult-testres/res_avg/di_avg.npy', avg_di)
#         np.save('bank-adult-testres/res_avg/spd_avg.npy', avg_spd)
#
#     print('avg calculating done. ')

