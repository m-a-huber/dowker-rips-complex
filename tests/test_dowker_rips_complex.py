import numpy as np
import pytest  # type: ignore
from sklearn.exceptions import NotFittedError  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from dowker_rips_complex import DowkerRipsComplex  # type: ignore

rng = np.random.default_rng(42)


@pytest.fixture
def random_data():
    n, dim = 500, 512
    ratio_vertices = 0.9
    X, y = (
        list(
            train_test_split(
                rng.standard_normal(size=(n, dim)), train_size=ratio_vertices
            )
        ),
        None,
    )
    return X, y


@pytest.fixture
def quadrilateral():
    vertices = np.array([[0, 0], [2, 0], [4, 2], [0, 4]])
    witnesses = np.array([[2, 3], [0, 2], [1, 0], [3, 1]])
    X, y = [vertices, witnesses], None
    return X, y


@pytest.fixture
def octagon():
    t = 1 / np.sqrt(2)
    vertices = np.array([[1, 0], [t, t], [0, 1], [-t, t]])
    witnesses = np.array([[-1, 0], [-t, -t], [0, -1], [t, -t]])
    X, y = [vertices, witnesses], None
    return X, y


def test_dowker_rips_complex(random_data):
    """
    Check whether `DowkerRipsComplex` runs at all for `max_dimension` up to and
    including `1`.
    """
    X, y = random_data
    for max_dimension in [0, 1]:
        drc = DowkerRipsComplex(max_dimension=max_dimension)
        drc.fit_transform(X, y)
        assert hasattr(drc, "persistence_")


def test_dowker_rips_complex_cosine(random_data):
    """
    Check whether `DowkerRipsComplex` runs on random data with non-default
    metric.
    """
    X, y = random_data
    drc = DowkerRipsComplex(metric="cosine")
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")


def test_dowker_rips_complex_not_fitted_error(random_data):
    """
    Check that `DowkerRipsComplex` raises a `NotFittedError` exception when
    calling `transform` without calling `fit` first on random data.
    """
    X, y = random_data
    drc = DowkerRipsComplex()
    with pytest.raises(NotFittedError):
        drc.transform(X)


def test_dowker_rips_complex_separate_calls(random_data):
    """
    Check whether `DowkerRipsComplex` runs on random data when `fit` and
    `transform` are called separately.
    """
    X, y = random_data
    drc = DowkerRipsComplex()
    drc.fit(X, y)
    drc.transform(X)
    assert hasattr(drc, "persistence_")


def test_dowker_rips_complex_empty_vertices():
    """
    Check whether `DowkerRipsComplex` runs for empty set of vertices and yields
    correct result.
    """
    X, y = (
        [
            rng.standard_normal(size=(0, 512)),
            rng.standard_normal(size=(10, 512)),
        ],
        None,
    )
    drc = DowkerRipsComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (drc.persistence_[0] == np.empty((0, 2))).all()
    assert (drc.persistence_[1] == np.empty((0, 2))).all()


def test_dowker_rips_complex_empty_witnesses():
    """
    Check whether `DowkerRipsComplex` runs for empty set of witnesses.
    """
    X, y = (
        [
            rng.standard_normal(size=(10, 512)),
            rng.standard_normal(size=(0, 512)),
        ],
        None,
    )
    drc = DowkerRipsComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (drc.persistence_[0] == np.empty((0, 2))).all()
    assert (drc.persistence_[1] == np.empty((0, 2))).all()


def test_dowker_rips_complex_empty_witnesses_no_swap():
    """
    Check whether `DowkerRipsComplex` runs for empty set of witnesses with
    `swap=False`.
    """
    X, y = (
        [
            rng.standard_normal(size=(10, 512)),
            rng.standard_normal(size=(0, 512)),
        ],
        None,
    )
    drc = DowkerRipsComplex(swap=False)
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (drc.persistence_[0] == np.empty((0, 2))).all()
    assert (drc.persistence_[1] == np.empty((0, 2))).all()


def test_dowker_rips_complex_quadrilateral(quadrilateral):
    """
    Check whether `DowkerRipsComplex` returns correct result on small
    quadrilateral.
    """
    drc = DowkerRipsComplex()
    drc.fit_transform(*quadrilateral)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (
        drc.persistence_[0] == np.array([[1, np.inf]], dtype=np.float32)
    ).all()
    assert (
        drc.persistence_[1]
        == np.array([[np.sqrt(5), np.sqrt(8)]], dtype=np.float32)
    ).all()


def test_dowker_rips_complex_octagon(octagon):
    """
    Check whether `DowkerRipsComplex` returns correct result on regular
    octagon.
    """
    drc = DowkerRipsComplex()
    drc.fit_transform(*octagon)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    birth = np.sqrt(2 - np.sqrt(2))
    death = np.sqrt(2 + np.sqrt(2))
    assert (
        drc.persistence_[0]
        == np.array([[birth, death], [birth, np.inf]], dtype=np.float32)
    ).all()
    assert (
        drc.persistence_[1] == np.empty(shape=(0, 2)).astype(np.float32)
    ).all()
