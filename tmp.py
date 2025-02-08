import numpy as np
from sklearn.metrics import pairwise_distances  # type: ignore


def get_dm(V, W):
    return pairwise_distances(V, W)


def custom_metric(X, Y):
    return np.min(
        np.maximum(X, Y)
    )


def get_ripser_input(dm):
    return pairwise_distances(dm, metric=custom_metric)


def gold(dm):
    return np.min(
        np.maximum(
            dm[:, :, None],
            dm.T[None, :, :]
        ),
        axis=1,
    )


if __name__ == "__main__":
    # n = 1000
    # ratio_vertices = 0.9
    # X = np.random.randn(n, 512)
    n = 100
    ratio_vertices = 0.9
    X = np.random.randn(n, 64)
    V, W = X[:int(ratio_vertices * n)], X[int(ratio_vertices) * n:]
    dm = get_dm(V, W)
    ripser_input = get_ripser_input(dm)
    assert (ripser_input == gold(dm)).all()
    print("Success.")
