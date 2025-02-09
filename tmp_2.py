import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split


def ripser_input_numpy(V, W):
    dm = pairwise_distances(V, W)
    ripser_input = np.min(
        np.maximum(
            dm[:, :, None],
            dm.T[None, :, :]
        ),
        axis=1,
    )
    return ripser_input

def ripser_input_sklearn(V, W):
    def custom_metric(a, b):
        return np.min(np.maximum(a, b))
    return pairwise_distances(
        pairwise_distances(V, W), metric=custom_metric
    )

def ripser_input_scipy(V, W):
    def custom_metric(a, b):
        return np.min(np.maximum(a, b))
    return cdist(
        cdist(V, W), cdist(V, W), metric=custom_metric
    )

def ripser_input_scipy_2(V, W):
    def custom_metric(a, b):
        return np.min(np.maximum(a, b))
    aux = cdist(V, W)
    res = squareform(pdist(
        aux, metric=custom_metric
    ))
    np.fill_diagonal(
        res,
        np.min(aux, axis=1)
    )
    return res


X = np.random.randn(1000, 512)
V, W = train_test_split(X, train_size=0.9)

%timeit res_0 = ripser_input_numpy(V, W)
# res_0.shape
# res_0

%timeit res_1 = ripser_input_sklearn(V, W)
# res_1.shape
# res_1

%timeit res_2 = ripser_input_scipy(V, W)
# res_2.shape
# res_2

%timeit res_3 = ripser_input_scipy_2(V, W)
# res_3.shape
# res_3

np.isclose(res_0, res_1).all(), np.isclose(res_0, res_2).all(), np.isclose(res_0, res_3).all()
