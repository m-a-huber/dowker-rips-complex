import numpy as np
import pytest  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from drips_complex import DripsComplex  # type: ignore


@pytest.fixture
def data_2_dim():
    n, dim = 500, 2
    ratio_vertices = 0.9
    X, y = list(
        train_test_split(
            np.random.randn(n, dim),
            train_size=ratio_vertices
        )
    ), None
    return X, y


@pytest.fixture
def data_3_dim():
    n, dim = 500, 3
    ratio_vertices = 0.9
    X, y = list(
        train_test_split(
            np.random.randn(n, dim),
            train_size=ratio_vertices
        )
    ), None
    return X, y


@pytest.fixture
def data_high_dim():
    n, dim = 500, 512
    ratio_vertices = 0.9
    X, y = list(
        train_test_split(
            np.random.randn(n, dim),
            train_size=ratio_vertices
        )
    ), None
    return X, y


@pytest.fixture
def quadrilateral():
    vertices = np.array([
        [0, 0],
        [2, 0],
        [4, 2],
        [0, 4]
    ])
    witnesses = np.array([
        [2, 3],
        [0, 2],
        [1, 0],
        [3, 1]
    ])
    X, y = [vertices, witnesses], None
    return X, y


@pytest.fixture
def octagon():
    t = 1 / np.sqrt(2)
    vertices = np.array([
        [1, 0],
        [t, t],
        [0, 1],
        [-t, t]
    ])
    witnesses = np.array([
        [-1, 0],
        [-t, -t],
        [0, -1],
        [t, -t]
    ])
    X, y = [vertices, witnesses], None
    return X, y


def test_drips_complex(data_high_dim):
    """
    Check whether `DripsComplex` runs at all and plots.
    """
    drc = DripsComplex()
    drc.fit_transform(*data_high_dim)
    assert hasattr(drc, "persistence_")


def test_drips_complex_plotting_2d(data_2_dim):
    """
    Check whether `DripsComplex` plots 2D data.
    """
    drc = DripsComplex()
    drc.fit_transform(*data_2_dim)
    assert hasattr(drc, "persistence_")
    drc.plot_points()
    drc.plot_persistence()


def test_drips_complex_plotting_3d(data_3_dim):
    """
    Check whether `DripsComplex` plots 3D data.
    """
    drc = DripsComplex()
    drc.fit_transform(*data_3_dim)
    assert hasattr(drc, "persistence_")
    drc.plot_points()
    drc.plot_persistence()


def test_drips_complex_quadrilateral(quadrilateral):
    """
    Check whether `DripsComplex` returns correct result on small quadrilateral.
    """
    drc = DripsComplex()
    drc.fit_transform(*quadrilateral)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (
        drc.persistence_[0] == np.array(
            [[1, np.inf]],
            dtype=np.float32
        )
    ).all()
    assert (
        drc.persistence_[1] == np.array(
            [[np.sqrt(5), np.sqrt(8)]],
            dtype=np.float32
        )
    ).all()


def test_drips_complex_octagon(octagon):
    """
    Check whether `DripsComplex` returns correct result on regular octagon.
    """
    drc = DripsComplex()
    drc.fit_transform(*octagon)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    birth = np.sqrt(2 - np.sqrt(2))
    death = np.sqrt(2 + np.sqrt(2))
    assert (
        drc.persistence_[0] == np.array([
            [birth, death],
            [birth, np.inf]
        ], dtype=np.float32)
    ).all()
    assert (
        drc.persistence_[1] == np.empty(shape=(0, 2)).astype(np.float32)
    ).all()


test_drips_complex_plotting_2d(data_2_dim())
