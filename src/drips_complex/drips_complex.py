from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as gobj  # type: ignore
from gph import ripser_parallel  # type: ignore
from numba import jit, prange  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore
from typing_extensions import Self

from .plotting.persistence_plotting import plot_persistences  # type: ignore
from .plotting.point_cloud_plotting import plot_point_cloud  # type: ignore


class DripsComplex(TransformerMixin, BaseEstimator):
    """ Class implementing the Dowker-Rips complex associated to a point cloud
    whose elements are separated into two classes. The data points on which the
    simplicial complex is constructed are referred to as "vertices", while the
    other ones are referred to as "witnesses".

    Parameters:
        max_dimension (int, optional): The maximum homology dimension computed.
            Will compute all dimensions lower than or equal to this value.
            Defaults to `1`.
        max_filtration (float, optional): The Maximum value of the Drips
            filtration parameter. If `np.inf`, the entire filtration is
            computed. Defaults to `np.inf`.
        coeff (int, optional): The field coefficient used in the computation of
            homoology. Defaults to 2.
        metric (str, optional): The metric used to compute distance between
            data points. Must be one of the metrics listed in
            ``sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
            Defaults to `"euclidean"`.
        metric_params (dict, optional): Additional parameters to be passed to
            the distance function. Defaults to `dict()`.
        verbose (bool, optional): Whether or not to display print some progress
            during fitting. Defaults to `False`.

    Attributes:
        vertices_ (numpy.ndarray of shape (n_vertices, dim)): NumPy-array
            containing the vertices.
        witnesses_ (numpy.ndarray of shape (n_witnesses, dim)): NumPy-array
            containing the witnesses.
        persistence_ (list[numpy.ndarray]): The persistent homology computed
            from the Drips simplicial complex. The format of this data is a
            list of NumPy-arrays of shape `(n_generators, 2)`, where the i-th
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
        verbose: bool = False,
    ) -> None:
        self.max_dimension = max_dimension
        self.coeff = coeff
        self.max_filtration = max_filtration
        self.metric = metric
        self.metric_params = metric_params
        self.verbose = verbose

    def vprint(
        self,
        s: str,
    ) -> None:
        if self.verbose:
            print(s)
        else:
            pass
        return

    def fit_transform(
        self,
        X: list[npt.NDArray],
        y: Optional[None] = None,
        swap: bool = False,
        n_threads: int = 1,
    ) -> Self:
        """Method that fits an `DripsComplex`-instance to a pair of point
        clouds consisting of vertices and witnesses and computes the persistent
        homology of the associated Dowker-Rips complex.

        Args:
            X: List containing the NumPy-arrays of vertices and witnesses, in
                this order.
            y: Not used, present here for API consistency with scikit-learn.
            swap: Whether or not to potentially swap the roles of vertices and
                witnesses to compute the less expensive variant of persistent
                homology. Defaults to `False`.
            n_threads (int, optional): Maximum number of threads to be used
                during the computation in homology dimensions 1 and above. `-1`
                means that the maximum number of threads will be used if
                possible. Defaults to `1`.

        Returns:
            :class:`drips_complex.DripsComplex`: Fitted instance of
                `DripsComplex`.
        """
        vertices, witnesses = X
        if vertices.shape[1] != witnesses.shape[1]:
            raise ValueError(
                "The vertices and witnesses should be of the same "
                f"dimensionality; received dim(vertices)={vertices.shape[1]} "
                f"and dim(witnesses)={witnesses.shape[1]}."
            )
        self.swap_ = swap
        if self.swap_ and len(vertices) > len(witnesses):
            vertices, witnesses = witnesses, vertices
            self.vprint("Swapped roles of vertices and witnesses.")
        self.vertices_ = vertices
        self.witnesses_ = witnesses
        self._labels_vertices_ = np.zeros(len(self.vertices_))
        self._labels_witnesses_ = -np.ones(len(self.witnesses_))
        self._points_ = np.concatenate([self.vertices_, self.witnesses_])
        self._labels_ = np.concatenate([
            self._labels_vertices_,
            self._labels_witnesses_
        ])
        self.vprint("Getting ripser input...")
        self._ripser_input_ = self._get_ripser_input()
        self.vprint("Done getting ripser input.")
        self.vprint("Computing persistent homology...")
        self.persistence_ = ripser_parallel(
            X=self._ripser_input_,
            metric="precomputed",
            maxdim=self.max_dimension,
            thresh=self.max_filtration,
            collapse_edges=True,
            n_threads=n_threads,
        )["dgms"]
        self.vprint("Done computing persistent homology.")
        return self.persistence_

    def _get_ripser_input(
        self,
    ):
        @jit(nopython=True, parallel=True)
        def _ripser_input_numba(dm):
            n = dm.shape[0]
            ripser_input = np.empty((n, n), dtype=np.float32)
            for i in prange(n):
                for j in range(i, n):
                    dist = np.min(np.maximum(dm[i], dm[j]))
                    ripser_input[i, j] = dist
                    ripser_input[j, i] = dist
            return ripser_input
        return _ripser_input_numba(
            pairwise_distances(
                X=self.vertices_,
                Y=self.witnesses_,
                metric=self.metric,
                **self.metric_params,
            )
        )

    def plot_persistence(
        self,
        **plotting_kwargs,
    ) -> gobj.Figure:
        """Method plotting the persistent homology of a Dowker-Rips complex.
        Underlying instance of `DripsComplex` must be fitted and have the
        attribute `persistence_`.

        Args:
            plotting_kwargs (optional): Keyword arguments passed to the
                function `plotting.persistence_plotting.plot_persistences`,
                such as `marker_size`.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: A plot of the
                persistence diagram.
        """
        if not hasattr(self, "persistence_"):
            raise AttributeError(
                "This instance does not have the attribute `persistence_`. "
                "Run `fit_transform` before plotting."
            )
        fig = plot_persistences(
            [self.persistence_],
            **plotting_kwargs,
        )
        return fig

    def plot_points(
        self,
        indicate_witnesses: bool = True,
        use_colors: bool = True,
        **plotting_kwargs,
    ) -> gobj.Figure:
        """Method plotting the vertices and witnesses of a Dowker-Rips complex.
        Underlying instance of `DripsComplex` must be fitted and have the
        attributes `vertices_` and `witnesses_`. Works for point clouds up to
        dimension three only.

        Args:
            indicate_witnesses (bool, optional): Whether or not to indicate the
                witness points by a cross as opposed to a dot.
                Defaults to `True`.
            use_colors (bool, optional): Whether or not to color the vertices
                and witnesses in different colors. Defaults to `True`.
            plotting_kwargs (optional): Keyword arguments passed to the
                function `plotting.point_cloud_plotting.plot_point_cloud`, such
                as `marker_size` and `colorscale`.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: A plot of the
                vertex and witness point clouds.
        """
        if not hasattr(self, "persistence_"):
            raise AttributeError(
                "This instance does not have the attributes `vertices_` and "
                "`witnesses_`. Run `fit_transform` before plotting."
            )
        if self._points_.shape[1] not in {1, 2, 3}:
            raise Exception(
                "Plotting is supported only for data "
                "sets of dimension at most 3."
            )
        return plot_point_cloud(
            self._points_,
            labels=self._labels_,
            indicate_outliers=indicate_witnesses,
            indicate_labels=use_colors,
            colorscale="wong",
            **plotting_kwargs,
        )
