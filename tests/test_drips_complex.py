import numpy as np
import pytest  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from drips_complex import DripsComplex  # type: ignore


@pytest.fixture
def random_data():
    n, dim = 500, 512
    ratio_vertices = 0.9
    X, y = (
        list(train_test_split(
            np.random.randn(n, dim), train_size=ratio_vertices)
        ),
        None,
    )
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


def test_drips_complex(random_data):
    """
    Check whether `DripsComplex` runs at all and plots.
    """
    X, y = random_data
    drc = DripsComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")


def test_drips_complex_empty_vertices():
    """
    Check whether `DripsComplex` runs for empty set of vertices.
    """
    X, y = [np.random.randn(0, 512), np.random.randn(10, 512)], None
    drc = DripsComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (
        drc.persistence_[0] == np.empty(
            (0, 2)
        )
    ).all()
    assert (
        drc.persistence_[1] == np.empty(
            (0, 2)
        )
    ).all()


def test_drips_complex_empty_witnesses():
    """
    Check whether `DripsComplex` runs for empty set of witnesses.
    """
    X, y = [np.random.randn(10, 512), np.random.randn(0, 512)], None
    drc = DripsComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (
        drc.persistence_[0] == np.empty(
            (0, 2)
        )
    ).all()
    assert (
        drc.persistence_[1] == np.empty(
            (0, 2)
        )
    ).all()


def test_drips_complex_plotting_2d(random_data):
    """
    Check whether `DripsComplex` plots 2D data.
    """
    X, y = random_data
    X = [pt_cloud[:, :2] for pt_cloud in X]
    drc = DripsComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    drc.plot_points()
    drc.plot_persistence()


def test_drips_complex_plotting_3d(random_data):
    """
    Check whether `DripsComplex` plots 3D data.
    """
    X, y = random_data
    X = [pt_cloud[:, :3] for pt_cloud in X]
    drc = DripsComplex()
    drc.fit_transform(X, y)
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
