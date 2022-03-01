import xlwt as wt
import numpy as np
import time


if __name__ == "__main__":
    # acc_array = np.load('./test_accu20220220145027.npy', allow_pickle=True)
    id_list = ['01', '02', '03', '04', '05']
    id_list_cnt = 0

    workbook = wt.Workbook()
    while id_list_cnt < 5:
        # di_array = np.load('res_avg/di_avg.npy')
        # spd_array = np.load('res_avg/spd_avg.npy')
        # di_array = np.load('bank-testres/di_res' + id_list[id_list_cnt] + '.npy')
        # spd_array = np.load('bank-testres/spd_res' + id_list[id_list_cnt] + '.npy')
        eoop_array = np.load('bank-testres/eoop_res' + id_list[id_list_cnt] + '.npy')
        eood_array = np.load('bank-testres/eood_res' + id_list[id_list_cnt] + '.npy')
        sheet = workbook.add_sheet('eoop_and_eood' + id_list[id_list_cnt])
        sheet.write(0, 0, 'test id')
        sheet.write(1, 0, 'equality_of_oppo')
        sheet.write(2, 0, 'equality_of_odds')
        for i in range(20):
            sheet.write(0, i + 1, i + 1)
            sheet.write(1, i + 1, str(eoop_array[i]))
            sheet.write(2, i + 1, str(eood_array[i]))
        id_list_cnt += 1
    workbook.save('./bank-testres/xls_file/eoop_eood_for_01_test.xls')
    print('test res saved as xls file.')
