import xlwt as wt
import numpy as np
import time


if __name__ == "__main__":
    # acc_array = np.load('./test_accu20220220145027.npy', allow_pickle=True)
    id_list = ['01', '02', '03', '04', '05']
    id_list_cnt = 0
    # di_array = np.load('res_avg/di_avg.npy')
    # spd_array = np.load('res_avg/spd_avg.npy')
    workbook = wt.Workbook()
    while id_list_cnt < 5:
        di_array = np.load('testres/di_res' + id_list[id_list_cnt] + '.npy')
        spd_array = np.load('testres/spd_res' + id_list[id_list_cnt] + '.npy')
        sheet = workbook.add_sheet('spd_and_di' + id_list[id_list_cnt])
        sheet.write(0, 0, 'test id')
        sheet.write(1, 0, 'disparate_impact')
        sheet.write(2, 0, 'statistical_parity_difference')
        for i in range(20):
            sheet.write(0, i + 1, i + 1)
            sheet.write(1, i + 1, str(di_array[i]))
            sheet.write(2, i + 1, str(spd_array[i]))
        id_list_cnt += 1
    workbook.save('xls_file/raw_group_fairness_in_5_tests.xls')
    print('test res saved as xls file.')
