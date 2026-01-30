import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from drc import DowkerRipsComplex

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
    X, _ = random_data
    drc = DowkerRipsComplex()
    with pytest.raises(NotFittedError):
        drc.transform(X)


def test_dowker_rips_complex_collapse_edges_and_return_generators(random_data):
    """
    Check that `DowkerRipsComplex` raises a `ValueError` exception when
    both `collapse_edges` and `return_generators` are set to True.
    """
    X, y = random_data
    drc = DowkerRipsComplex(collapse_edges=True, return_generators=True)
    with pytest.raises(ValueError):
        drc.fit(X, y)


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


def test_dowker_rips_complex_fit_wrong_number_of_arrays():
    """
    Check that `fit` raises ValueError when X does not contain exactly 2
    arrays.
    """
    drc = DowkerRipsComplex()
    X_one = [rng.standard_normal(size=(10, 2))]
    with pytest.raises(ValueError):
        drc.fit(X_one)
    X_three = [
        rng.standard_normal(size=(10, 2)),
        rng.standard_normal(size=(10, 2)),
        rng.standard_normal(size=(10, 2)),
    ]
    with pytest.raises(ValueError):
        drc.fit(X_three)


def test_dowker_rips_complex_fit_not_2d():
    """
    Check that `fit` raises ValueError when vertices or witnesses are not 2D.
    """
    drc = DowkerRipsComplex()
    X_1d = [
        rng.standard_normal(size=(10)),
        rng.standard_normal(size=(10)),
    ]
    with pytest.raises(ValueError):
        drc.fit(X_1d)
    X_3d = [
        rng.standard_normal(size=(10, 10, 10)),
        rng.standard_normal(size=(10, 10, 10)),
    ]
    with pytest.raises(ValueError):
        drc.fit(X_3d)


def test_dowker_rips_complex_fit_dimension_mismatch():
    """
    Check that `fit` raises ValueError when vertex and witness dimensions
    differ.
    """
    drc = DowkerRipsComplex()
    X = [
        rng.standard_normal(size=(10, 1)),
        rng.standard_normal(size=(10, 3)),
    ]
    with pytest.raises(ValueError):
        drc.fit(X)


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
    assert drc.persistence_[0].shape == (0, 2)
    assert drc.persistence_[1].shape == (0, 2)


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
    assert drc.persistence_[0].shape == (0, 2)
    assert drc.persistence_[1].shape == (0, 2)


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
    assert drc.persistence_[0].shape == (0, 2)
    assert drc.persistence_[1].shape == (0, 2)


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
    assert drc.persistence_[1].shape == (0, 2)
