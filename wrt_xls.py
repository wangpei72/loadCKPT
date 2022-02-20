import xlwt as wt
import numpy as np
import time


if __name__ == "__main__":
    acc_array = np.load('./test_accu20220220145027.npy', allow_pickle=True)
    workbook = wt.Workbook()
    sheet = workbook.add_sheet('sheet1')
    sheet.write(0, 0, 'test id')
    sheet.write(1, 0, 'test accuracy')
    for i in range(20):
        sheet.write(0, i + 1, i + 1)
        sheet.write(1, i + 1, acc_array[i])

    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    workbook.save('test' + time_str + '.xls')
    print('test accu res saved as xls file.')
