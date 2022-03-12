import xlwt as wt
import numpy as np
import time
import sys

if __name__ == "__main__":
    # acc_array = np.load('./test_accu20220220145027.npy', allow_pickle=True)
    id_list = ['avg', '02', '03', '04', '05']
    id_list_cnt = 0

    workbook = wt.Workbook()
    while id_list_cnt < 1:
        di_array = np.load('adult-testres-age/res_avg/di_avg.npy')
        spd_array = np.load('adult-testres-age/res_avg/spd_avg.npy')
        eoop_array = np.load('adult-testres-age/res_avg/eoop_avg.npy')
        eood_array = np.load('adult-testres-age/res_avg/eood_avg.npy')
        # acc_arry = np.load('bank-testres/res_avg/test_accu_avg.npy', allow_pickle=True)

        sheet = workbook.add_sheet('group_fair_adult' + id_list[id_list_cnt])
        sheet.write(0, 0, 'test id')
        sheet.write(1, 0, 'di')
        sheet.write(2, 0, 'spd')
        sheet.write(3, 0, 'eoop')
        sheet.write(4, 0, 'eood')
        # sheet.write(3, 0, 'predict_accuracy')
        for i in range(20):
            sheet.write(0, i + 1, i + 1)
            sheet.write(1, i + 1, str(di_array[i]))
            sheet.write(2, i + 1, str(spd_array[i]))
            sheet.write(3, i + 1, str(eoop_array[i]))
            sheet.write(4, i + 1, str(eood_array[i]))
            # sheet.write(3, i + 1, str(acc_arry[i]))
        id_list_cnt += 1
    workbook.save('adult-testres-age/xls_file/group_fair_adult_age.xls')
    print('test res saved as xls file.')


# if __name__ == "__main__":
#     # acc_array = np.load('./test_accu20220220145027.npy', allow_pickle=True)
#     id_list = ['avg', '02', '03', '04', '05']
#     id_list_cnt = 0
#
#     workbook = wt.Workbook()
#     while id_list_cnt < 1:
#         eoop_array = np.load('bank-testres/res_avg/eoop_avg.npy')
#         eood_array = np.load('bank-testres/res_avg/eood_avg.npy')
#         acc_arry = np.load('bank-testres/res_avg/test_accu_avg.npy', allow_pickle=True)
#         # di_array = np.load('bank-adult-testres/di_res' + id_list[id_list_cnt] + '.npy')
#         # spd_array = np.load('bank-adult-testres/spd_res' + id_list[id_list_cnt] + '.npy')
#         # eoop_array = np.load('bank-adult-testres/eoop_res' + id_list[id_list_cnt] + '.npy')
#         # eood_array = np.load('bank-adult-testres/eood_res' + id_list[id_list_cnt] + '.npy')
#
#         sheet = workbook.add_sheet('eoop_and_eood' + id_list[id_list_cnt])
#         sheet.write(0, 0, 'test id')
#         sheet.write(1, 0, 'equality_of_oppo')
#         sheet.write(2, 0, 'equality_of_odds')
#         sheet.write(3, 0, 'predict_accuracy')
#         for i in range(20):
#             sheet.write(0, i + 1, i + 1)
#             sheet.write(1, i + 1, str(eoop_array[i]))
#             sheet.write(2, i + 1, str(eood_array[i]))
#             sheet.write(3, i + 1, str(acc_arry[i]))
#         id_list_cnt += 1
#     workbook.save('./bank-adult-testres/xls_file/bank_eoop_eood_avg_results.xls')
#     print('test res saved as xls file.')
