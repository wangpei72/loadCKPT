import numpy as np


default_map = ['Age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
def compute_boolean_conditioning_vector(X, feature_names=None, condition=None):
    """
    condition (list(dict))
    Examples:
        >>> condition = [{'sex': 1, 'age': 1}, {'sex': 0}]

        This corresponds to `(sex == 1 AND age == 1) OR (sex == 0)`.
    """
    if feature_names is None:
        feature_names = default_map
    if condition is None:
        return np.ones(X.shape[0], dtype=bool)

    overall_cond = np.zeros(X.shape[0], dtype=bool)
    for group in condition:
        group_cond = np.ones(X.shape[0], dtype=bool)
        for name, val in group.items():
            index = feature_names.index(name)
            group_cond = np.logical_and(group_cond, X[:, index] == val)
        overall_cond = np.logical_or(overall_cond, group_cond)

    return overall_cond


def compute_num_instances(X, w, feature_names=None, condition=None):
    """Compute the number of instances, :math:`n`, conditioned on the protected
    attribute(s).

    Args:
        X (numpy.ndarray): Dataset features.
        w (numpy.ndarray): Instance weight vector.
        feature_names (list): Names of the features.
        condition (list(dict)): Same format as
            :func:`compute_boolean_conditioning_vector`.

    Returns:
        int: Number of instances (optionally conditioned).
    """

    # condition if necessary
    if feature_names is None:
        feature_names = default_map
    cond_vec = compute_boolean_conditioning_vector(X, feature_names, condition)

    return np.sum(w[cond_vec], dtype=np.float64)


def get_subset_by_protected_attr(X, privileged=True):
    condition_boolean_vector = compute_boolean_conditioning_vector(X, condition=[{'sex': 1}])
