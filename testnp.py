import numpy as np
import sys

if __name__ == '__main__':
    a = np.array([1, 1, 0], dtype=np.int32)
    sample = np.load('./testres/di_res01.npy')
    print(sample[0])
