import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
from gph import ripser_parallel  # type: ignore
from numba import jit, prange  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore


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
            homoology. Defaults to `2`.
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
            containing the vertices.
        witnesses_ (numpy.ndarray of shape (n_witnesses, dim)): NumPy-array
            containing the witnesses.
        persistence_ (list[numpy.ndarray]): The persistent homology computed
            from the Dowker-Rips simplicial complex. The format of this data is
            a list of NumPy-arrays of shape `(n_generators, 2)`, where the i-th
            entry of the list is an array containing the birth and death times
            of the homological generators in dimension i-1. In particular, the
            list starts with 0-dimensional homology and contains information
            from consecutive homological dimensions.
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

    def fit_transform(
        self,
        X: list[npt.NDArray],
        y: Optional[None] = None,
    ) -> list[npt.NDArray]:
        """Method that fits a `DowkerRipsComplex`-instance to a pair of point
        clouds consisting of vertices and witnesses by computing the persistent
        homology of the associated Dowker-Rips complex.

        Args:
            X (list[numpy.ndarray]): List containing the NumPy-arrays of
                vertices and witnesses, in this order.
            y (None, optional): Not used, present here for API consistency with
                scikit-learn.

        Returns:
            list[numpy.ndarray]: The persistent homology computed from the
                Dowker-Rips simplicial complex. The format of this data is a
                list of NumPy-arrays of shape `(n_generators, 2)`, where the
                i-th entry of the list is an array containing the birth and
                death times of the homological generators in dimension i-1. In
                particular, the list starts with 0-dimensional homology and
                contains information from consecutive homological dimensions.
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
            "Complex has (n_vertices, n_witnesses) = "
            f"{(len(self.vertices_), len(self.witnesses_))}."
        )
        self._labels_vertices_ = np.zeros(len(self.vertices_))
        self._labels_witnesses_ = -np.ones(len(self.witnesses_))
        self._points_ = np.concatenate([self.vertices_, self.witnesses_])
        self._labels_ = np.concatenate(
            [self._labels_vertices_, self._labels_witnesses_]
        )
        if min(len(self.vertices_), len(self.witnesses_)) == 0:
            self._ripser_input_ = np.empty((0, 0))
            self.persistence_ = [
                np.empty((0, 2), dtype=np.float32)
                for _ in range(self.max_dimension + 1)
            ]
        else:
            self.vprint("Getting ripser input...")
            self._ripser_input_ = self._get_ripser_input()
            self.vprint(
                "Done getting ripser input, has shape "
                f"{self._ripser_input_.shape}."
            )
            self.vprint("Computing persistent homology...")
            self.persistence_ = ripser_parallel(
                X=self._ripser_input_,
                metric="precomputed",
                maxdim=self.max_dimension,
                thresh=self.max_filtration,
                coeff=self.coeff,
                collapse_edges=self.collapse_edges,
                n_threads=self.n_threads,
            )["dgms"]
            self.vprint("Done computing persistent homology.")
        return self.persistence_

    def _get_ripser_input(
        self,
    ):
        self._dm_ = pairwise_distances(
            X=self.vertices_,
            Y=self.witnesses_,
            metric=self.metric,
            **self.metric_params,
        )
        if self.use_numpy:
            try:
                return np.min(
                    np.maximum(
                        self._dm_.T[:, :, None],
                        self._dm_[None, :, :]
                    ),
                    axis=1,
                )
            except MemoryError:
                warnings.warn(
                    "NumPy implementation ran out of memory; "
                    "falling back to Numba.",
                    RuntimeWarning
                )

        @jit(nopython=True, parallel=True)
        def _ripser_input_numba(dm):
            n = dm.shape[0]
            ripser_input = np.empty((n, n))
            for i in prange(n):
                for j in range(i, n):
                    dist = np.min(np.maximum(dm[i], dm[j]))
                    ripser_input[i, j] = dist
                    ripser_input[j, i] = dist
            return ripser_input
        return _ripser_input_numba(
            self._dm_
        )
