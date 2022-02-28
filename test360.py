import sys
sys.path.append("./")
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import tensorflow as tf
tf.disable_eager_execution()

if __name__ == '__main__':
    dataset_origin = load_preproc_data_adult()

    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    dataset_origin_train, dataset_origin_test = dataset_origin.split([0.8], shuffle=True)
    print(dataset_origin_train.protected_attribute_names)
    print(dataset_origin_train.feature_names)



