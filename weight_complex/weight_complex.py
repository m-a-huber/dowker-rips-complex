from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.metrics import pairwise_distances
from gph import ripser_parallel
from datasets_custom.plotting import plot_point_cloud, plot_persistences


class WeightComplex(BaseEstimator):
    def __init__(
        self,
        metric="euclidean",
        p=2,
        max_dimension=2,
        max_filtration=np.inf
    ):
        self.metric = metric
        self.p = p
        self.max_dimension = max_dimension
        self.max_filtration = max_filtration

    def fit(
        self,
        vertices,
        witnesses,
        n_threads=-1,
        **persistence_kwargs
    ):
        self.vertices_ = vertices
        self.witnesses_ = witnesses
        self._labels_vertices_ = np.zeros(len(self.vertices_))
        self._labels_witnesses_ = -np.ones(len(self.witnesses_))
        self._points_ = np.concatenate([self.vertices_, self.witnesses_])
        self._labels_ = np.concatenate([
            self._labels_vertices_,
            self._labels_witnesses_
        ])
        ###############
        self._ripser_input = self._get_ripser_input(
            self.vertices_,
            self.witnesses_
        )
        self.persistence_ = ripser_parallel(
            X=self._ripser_input,
            metric="precomputed",
            maxdim=self.max_dimension-1,
            thresh=self.max_filtration,
            collapse_edges=True,
            return_generators=False,
            n_threads=n_threads,
            **persistence_kwargs
        )["dgms"]
        ###############
        # self.weights_ = self._get_weights(
        #     self.vertices_,
        #     self.witnesses_
        # )
        # self.persistence_ = ripser_parallel(
        #     X=self.vertices_,
        #     weights=self.weights_,
        #     weight_params={"p": self.p},
        #     metric=self.metric,
        #     maxdim=self.max_dimension-1,
        #     thresh=self.max_filtration,
        #     collapse_edges=True,
        #     return_generators=False,
        #     n_threads=n_threads,
        #     **persistence_kwargs
        # )["dgms"]
        ###############
        return self

    def _get_ripser_input(
        self,
        vertices,
        witnesses
    ):
        ###############
        # midpts = 0.5 * (
        #     vertices[:, None, :] + vertices
        # )
        # diffs = midpts[None, :] - witnesses[:, None, None]
        # norms = np.linalg.norm(
        #     diffs,
        #     ord=self.p,
        #     axis=-1
        # )
        # ripser_input = np.min(norms, axis=0)
        # ripser_input[np.diag_indices_from(input)] = ripser_input.min(axis=1)
        ###############
        self._dm_ = pairwise_distances(
            self.witnesses_,
            self.vertices_,
            metric=self.metric
        )
        ripser_input = np.min(
            np.maximum(
                self._dm_.T[:, :, None],
                self._dm_[None, :, :]
            ),
            axis=1
        )
        ###############
        return ripser_input

    # def _get_weights(
    #     self,
    #     vertices,
    #     witnesses
    # ):
    #     self._dm_ = pairwise_distances(
    #         self.witnesses_,
    #         self.vertices_,
    #         metric=self.metric
    #     )
    #     return np.min(
    #         self._dm_,
    #         axis=0
    #     )

    def plot_persistence(self, **plotting_kwargs):
        check_is_fitted(self, attributes="persistence_")
        fig = plot_persistences(
            [self.persistence_],
            **plotting_kwargs
        )
        return fig

    def plot_points(
        self,
        indicate_witnesses=True,
        use_colors=True,
        **plotting_kwargs
    ):
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
            **plotting_kwargs
        )
