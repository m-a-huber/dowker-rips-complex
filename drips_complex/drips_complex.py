import numpy as np
import numpy.typing as npt
import plotly.graph_objects as gobj  # type: ignore
from datasets_custom.plotting import (  # type: ignore
    plot_persistences,
    plot_point_cloud,
)
from gph import ripser_parallel  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from typing_extensions import Self


class DripsComplex(BaseEstimator):
    """ #TODO

    Parameters:
        metric (str, optional): The metric used to compute distance between
            data points. Must be one of the metrics listed in
            ``sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
            Defaults to `"euclidean"`.
        max_dimension (int, optional): The maximum homology dimension computed.
            Will compute all dimensions lower than or equal to this value.
            Defaults to 1.
        max_filtration (float, optional): The Maximum value of the Drips
            filtration parameter. If `np.inf`, the entire filtration is
            computed. Defaults to `np.inf`.

    Attributes:
        vertices_ (numpy.ndarray of shape (n_vertices, dim)): NumPy-array
            containing the vertices.
        witnesses_ (numpy.ndarray of shape (n_witnesses, dim)): NumPy-array
            containing the witnesses.
        persistence_ (list[numpy.ndarray]): The persistent homology computed
            from the Drips simplicial complex. The format of this data is a
            list of NumPy-arrays of shape (n_generators, 2), where the i-th
            entry of the list is an array containing the birth and death times
            of the homological generators in dimension i-1. In particular, the
            list starts with 0-dimensional homology and contains information
            from consecutive homological dimensions.

    References:
        #TODO
    """

    def __init__(
        self,
        metric: str = "euclidean",
        max_dimension: int = 1,
        max_filtration: float = np.inf,
    ) -> None:
        self.metric = metric
        self.max_dimension = max_dimension
        self.max_filtration = max_filtration

    def fit(
        self,
        vertices: npt.NDArray,
        witnesses: npt.NDArray,
        n_threads: int = -1,
        precision: str = "double",
        swap: bool = False,
        **persistence_kwargs,
    ) -> Self:
        """Method that fits an Drips instance to a pair of point clouds
        consisting of vertices and witnesses.

        Args:
            vertices (numpy.ndarray of shape (n_vertices, dim)): NumPy-array
                containing the vertices.
            witnesses (numpy.ndarray of shape (n_witnesses, dim)): NumPy-array
                containing the witnesses.
            n_threads (int, optional): Maximum number of threads to be used
                during the computation in homology dimensions 1 and above. -1
                means that the maximum number of threads will be used if
                possible. Defaults to -1.
            precision (str, optional): Floating point precision to be used
                throughout computation. Must be one of `"double"`, `"single"`
                and `"half"`. Defaults to `"double"`
            swap (bool, optional): Whether or not to potentially swap the roles
                of vertices and witnesses to compute the less expensive variant
                of persistent homology (both are guaranteed to coincide).
                Defaults to False.

        Returns:
            :class:`drips_complex.DripsComplex`: Fitted instance of
                DripsComplex.
        """
        self.precision_ = precision
        match self.precision_:
            case "double":
                dtype = np.float64
            case "single":
                dtype = np.float32  # type: ignore
            case "half":
                dtype = np.float16  # type: ignore
        self.swap_ = swap
        if self.swap_ and len(vertices) > len(witnesses):
            vertices, witnesses = witnesses, vertices
        self.vertices_ = vertices.astype(dtype)
        self.witnesses_ = witnesses.astype(dtype)
        self._labels_vertices_ = np.zeros(len(self.vertices_))
        self._labels_witnesses_ = -np.ones(len(self.witnesses_))
        self._points_ = np.concatenate([self.vertices_, self.witnesses_])
        self._labels_ = np.concatenate([
            self._labels_vertices_,
            self._labels_witnesses_
        ])
        self._ripser_input_ = self._get_ripser_input(
            self.vertices_,
            self.witnesses_,
            dtype,
        )
        self.persistence_ = ripser_parallel(
            X=self._ripser_input_,
            metric="precomputed",
            maxdim=self.max_dimension,
            thresh=self.max_filtration,
            collapse_edges=True,
            return_generators=False,
            n_threads=n_threads,
            **persistence_kwargs,
        )["dgms"]
        return self

    def _get_ripser_input(
        self,
        vertices,
        witnesses,
        dtype,
    ):
        # Vertex wgts: min dist to W
        # Edge wgts: min_{w\in W}(max[d(v,w), d(v',w)])
        self._dm_ = pairwise_distances(
            self.witnesses_,
            self.vertices_,
            metric=self.metric,
        ).astype(dtype)
        ripser_input = np.min(
            np.maximum(
                self._dm_.T[:, :, None],
                self._dm_[None, :, :]
            ),
            axis=1,
        )
        return ripser_input

    def plot_persistence(
        self,
        **plotting_kwargs,
    ) -> gobj.Figure:
        """Method plotting the Drips persistence. Underlying instance must be
        fitted and have the attribute `persistence_`.

        Args:
            plotting_kwargs (optional): Arguments passed to the function
                `datasets_custom.persistence_plotting.plot_persistences`.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: A plot of the
                persistence diagram.
        """
        check_is_fitted(self, attributes="persistence_")
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
        """Method plotting the vertices and witnesses underlying a fitted
        instance of DripsComplex. Works for point clouds up to dimension three
        only.

        Args:
            indicate_witnesses (bool, optional): Whether or not to use a
                distinguished marker to indicate the witness points.
                Defaults to True.
            use_colors (bool, optional): Whether or not to color the vertices
                and witnesses in different colors. Defaults to True.
            plotting_kwargs (optional): Arguments passed to the function
                `datasets_custom.utils.plotting.plot_point_cloud`, such as
                `marker_size` and `colorscale`.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: A plot of the
                vertex and witness point clouds.
        """
        check_is_fitted(self, attributes=["vertices_", "witnesses_"])
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
