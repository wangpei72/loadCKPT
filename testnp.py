import numpy as np
import sys

if __name__ == '__main__':
    a = np.array([5, 1, 5], dtype=np.float32)
    b = np.array([6, 0, 54], dtype=np.float32)
    sample = np.load('adult-testres/di_res01.npy')
    sample2 = a / 5.
    a += b
    print(sample[0])
