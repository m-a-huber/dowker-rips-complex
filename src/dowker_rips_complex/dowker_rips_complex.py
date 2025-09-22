import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
from gph import ripser_parallel  # type: ignore
from numba import jit, prange  # type: ignore
from sklearn.base import (  # type: ignore
    BaseEstimator,
    TransformerMixin,
    check_is_fitted,
)
from sklearn.metrics import pairwise_distances


class DowkerRipsComplex(TransformerMixin, BaseEstimator):
    """Class implementing the Dowker-Rips persistent homology associated to a
    point cloud whose elements are separated into two classes. The data points
    on which the underlying simplicial complex is constructed are referred to
    as "vertices", while the other ones are referred to as "witnesses".

    Parameters:
        max_dimension (int, optional): The maximum homology dimension computed.
            Will compute all dimensions lower than or equal to this value.
            Defaults to `1`.
        max_filtration (float, optional): The Maximum value of the Dowker-Rips
            filtration parameter. If `np.inf`, the entire filtration is
            computed. Defaults to `np.inf`.
        coeff (int, optional): The field coefficient used in the computation of
            homology. Defaults to `2`.
        metric (str, optional): The metric used to compute distance between
            data points. Must be one of the metrics listed in
            ``sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
            Defaults to `"euclidean"`.
        metric_params (dict, optional): Additional parameters to be passed to
            the distance function. Defaults to `dict()`.
        use_numpy (bool, optional): Whether or not to use NumPy instead of
            Numba to compute the input to `ripser_parallel` from the matrix of
            pairwise distances. The Numba implementation does not suffer from
            OOM errors, and will be fallen back to if `use_numpy` is set to
            `True` and the NumPy implementation results in such an error.
            Defaults to `False`.
        collapse_edges (bool, optional): Whether to collapse edges prior to
            computing persistence in order to speed up that computation. Not
            recommended unless for very large datasets. Defaults to `False`.
        n_threads (int, optional): Maximum number of threads to be used during
            the computation in homology dimensions 1 and above. `-1` means that
            the maximum number of threads will be used if possible.
            Defaults to `1`.
        swap (bool, optional): Whether or not to potentially swap the roles of
            vertices and witnesses to compute the less expensive variant of
            persistent homology. Note that this may affect the resulting
            persistence in dimensions two and higher. Defaults to `True`.
        verbose (bool, optional): Whether or not to display information such as
            computation progress. Defaults to `False`.

    Attributes:
        vertices_ (numpy.ndarray of shape (n_vertices, dim)): NumPy-array
            containing the vertices. Note that setting `swap=True` may swap the
            roles of vertices and witnesses.
        witnesses_ (numpy.ndarray of shape (n_witnesses, dim)): NumPy-array
            containing the witnesses. Note that setting `swap=True` may swap
            the roles of vertices and witnesses.
        dm_ (numpy.ndarray of shape (n_vertices, n_witnesses)): NumPy-array
            containing the pairwise distances between vertices and witnesses.
            Note that setting `swap=True` may swap the roles of vertices and
            witnesses.
        ripser_input_ (numpy.ndarray of shape (n_vertices, n_vertices)): NumPy-
            array containing the custom distance matrix encoding the Dowker-
            Rips complex, which is subsequently used for the computation of
            persistent homology. Note that setting `swap=True` may swap the
            roles of vertices and witnesses.
        persistence_ (list[numpy.ndarray]): The persistent homology computed
            from the Dowker-Rips simplicial complex. The format of this data is
            a list of NumPy-arrays of dtype float32 and of shape
            `(n_generators, 2)`, where the i-th entry of the list is an array
            containing the birth and death times of the homological generators
            in dimension i-1. In particular, the list starts with 0-dimensional
            homology and contains information from consecutive homological
            dimensions. Only present if `transform` has been called.
    """

    def __init__(
        self,
        max_dimension: int = 1,
        max_filtration: float = np.inf,
        coeff: int = 2,
        metric: str = "euclidean",
        metric_params: dict = dict(),
        use_numpy: bool = False,
        collapse_edges: bool = False,
        n_threads: int = 1,
        swap: bool = True,
        verbose: bool = False,
    ) -> None:
        self.max_dimension = max_dimension
        self.max_filtration = max_filtration
        self.coeff = coeff
        self.metric = metric
        self.metric_params = metric_params
        self.use_numpy = use_numpy
        self.collapse_edges = collapse_edges
        self.n_threads = n_threads
        self.swap = swap
        self.verbose = verbose

    def vprint(
        self,
        s: str,
    ) -> None:
        if self.verbose:
            print(s)
        return

    def fit(
        self,
        X: list[npt.NDArray],
        y: Optional[None] = None,
    ) -> "DowkerRipsComplex":
        """Method that fits an instance of `DowkerRipsComplex` to a pair of
        point clouds consisting of vertices and witnesses. Computes the custom
        distance matrix used in the computation of Dowker-Rips persistent
        homology in a possible subsequent call to `transform`.

        Args:
            X (list[numpy.ndarray]): List containing the NumPy-arrays of
                vertices and witnesses, in this order.
            y (None, optional): Not used, present here for API consistency with
                scikit-learn.

        Returns:
            DowkerRipsComplex: Fitted instance of `DowkerRipsComplex`.

        Raises:
            ValueError: If the vertices and witnesses are not of the same
                dimensionality.
        """
        vertices, witnesses = X
        if vertices.shape[1] != witnesses.shape[1]:
            raise ValueError(
                "The vertices and witnesses should be of the same "
                f"dimensionality; received dim(vertices)={vertices.shape[1]} "
                f"and dim(witnesses)={witnesses.shape[1]}."
            )
        if self.swap and len(vertices) > len(witnesses):
            vertices, witnesses = witnesses, vertices
            self.vprint("Swapped roles of vertices and witnesses.")
        self.vertices_ = vertices
        self.witnesses_ = witnesses
        self.vprint(
            "Dowker-Rips complex has (n_vertices, n_witnesses) = "
            f"{(len(self.vertices_), len(self.witnesses_))}."
        )
        self._labels_vertices_ = np.zeros(len(self.vertices_))
        self._labels_witnesses_ = -np.ones(len(self.witnesses_))
        self._points_ = np.concatenate([self.vertices_, self.witnesses_])
        self._labels_ = np.concatenate(
            [self._labels_vertices_, self._labels_witnesses_]
        )
        if min(len(self.vertices_), len(self.witnesses_)) == 0:
            self.dm_ = np.empty((len(self.vertices_), len(self.witnesses_)))
            self.ripser_input_ = np.empty((0, 0))
        else:
            self.dm_ = pairwise_distances(
                X=self.vertices_,
                Y=self.witnesses_,
                metric=self.metric,
                **self.metric_params,
            )
            self.vprint("Computing `ripser_input`...")
            self.ripser_input_ = self._get_ripser_input()
            self.vprint(
                "Finished computing `ripser_input`, has shape "
                f"{self.ripser_input_.shape}."
            )
        return self

    def transform(
        self,
        X: list[npt.NDArray],
    ) -> list[npt.NDArray[np.float32]]:
        """Method that uses the underlying fitted instance of
        `DowkerRipsComplex` to compute the Dowker-Rips persistent homology of a
        pair point clouds consisting of vertices and witnesses.

        Args:
            X (list[numpy.ndarray]): List containing the NumPy-arrays of
                vertices and witnesses, in this order.

        Returns:
            list[numpy.ndarray]: The persistent homology computed from the
                Dowker-Rips simplicial complex. The format of this data is a
                list of NumPy-arrays of dtype float32 and of shape
                `(n_generators, 2)`, where the i-th entry of the list is an
                array containing the birth and death times of the homological
                generators in dimension i-1. In particular, the list starts
                with 0-dimensional homology and contains information from
                consecutive homological dimensions.

        Raises:
            sklearn.exceptions.NotFittedError: If `fit` has not yet been called
                on the underlying instance of `DowkerRipsComplex`.
        """
        check_is_fitted(self, attributes=["ripser_input_"])
        if min(len(self.vertices_), len(self.witnesses_)) == 0:
            self.persistence_ = [
                np.empty((0, 2), dtype=np.float32)
                for _ in range(self.max_dimension + 1)
            ]
        else:
            self.vprint("Computing persistent homology...")
            self.persistence_ = ripser_parallel(
                X=self.ripser_input_,
                metric="precomputed",
                maxdim=self.max_dimension,
                thresh=self.max_filtration,
                coeff=self.coeff,
                collapse_edges=self.collapse_edges,
                n_threads=self.n_threads,
            )["dgms"]
            self.vprint("Finished computing persistent homology.")
        return self.persistence_

    def _get_ripser_input(
        self,
    ):
        if self.use_numpy:
            try:
                return np.min(
                    np.maximum(self.dm_.T[:, :, None], self.dm_[None, :, :]),
                    axis=1,
                )
            except MemoryError:
                warnings.warn(
                    "NumPy implementation ran out of memory; "
                    "falling back to Numba.",
                    RuntimeWarning,
                )

        @jit(nopython=True, parallel=True)
        def ripser_input_numba(dm):
            n = dm.shape[0]
            ripser_input = np.empty((n, n))
            for i in prange(n):
                for j in range(i, n):
                    dist = np.min(np.maximum(dm[i], dm[j]))
                    ripser_input[i, j] = dist
                    ripser_input[j, i] = dist
            return ripser_input

        return ripser_input_numba(self.dm_)
