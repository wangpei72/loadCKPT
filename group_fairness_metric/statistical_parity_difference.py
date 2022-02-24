import numpy as np
from .utils import compute_boolean_conditioning_vector, compute_num_instances

map_label = ['<50k', '>50k']


def S_P_D(X, y):
    # 越接近0越公平
    unprivileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, condition=[{'sex': 0}])
    privileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, condition=[{'sex': 1}])
    unpri_instances_num = np.sum(unprivileged_boolean_condition_vector != 0, dtype=np.float32)
    pri_instances_num = np.sum(privileged_boolean_condition_vector != 0, dtype=np.float32)

    unpri_favourable_b_c_v = np.logical_and((y == 1), unprivileged_boolean_condition_vector)
    unpri_favourable_num = np.sum(unpri_favourable_b_c_v != 0, dtype=np.float32)
    prob_pos_unpri = unpri_favourable_num / unpri_instances_num

    pri_favourable_b_c_v = np.logical_and((y == 1), privileged_boolean_condition_vector)
    pri_favourable_num = np.sum(pri_favourable_b_c_v != 0, dtype=np.float32)
    prob_pos_pri = pri_favourable_num / pri_instances_num

    return prob_pos_unpri - prob_pos_pri





