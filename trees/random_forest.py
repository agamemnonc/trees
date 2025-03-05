from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.tree import BaseDecisionTree
from sklearn.base import BaseEstimator


class RandomForestBase(BaseEstimator, ABC):
    def __init__(
        self,
        tree: BaseDecisionTree,
        n_trees: int,
        max_samples: int,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.trees_ = []

    def fit(self, X: NDArray, y: NDArray) -> "RandomForestBase":
        n_samples, n_features = X.shape
        for _ in range(self.n_trees):
            if self.bootstrap:
                assert (
                    self.max_samples <= n_samples
                ), "max_samples must be less than or equal to n_samples when bootstrap is True"
                sample_indices = self.get_sample_indices(n_samples, _)
                X_boot = X[sample_indices]
                y_boot = y[sample_indices]
            else:
                X_boot = X
                y_boot = y

            tree = self.tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        pass

    def check_is_fitted(self):
        if len(self.trees_) != self.n_trees:
            raise ValueError("RandomForestBase is not fitted yet")

    def get_sample_indices(self, n_samples: int, iterator_idx: int) -> NDArray:
        # Create a new random state for each tree using the base random_state and iterator_idx
        tree_random_state = np.random.RandomState(
            self.random_state + iterator_idx
            if self.random_state is not None
            else iterator_idx
        )
        sample_indices = tree_random_state.choice(
            n_samples, size=self.max_samples, replace=False
        )
        return sample_indices

    def get_oob_indices(self, n_samples: int, iterator_idx: int) -> NDArray:
        sample_indices = self.get_sample_indices(n_samples, iterator_idx)
        return np.setdiff1d(np.arange(n_samples), sample_indices)


class RandomForestClassifier(RandomForestBase):
    def __init__(
        self,
        tree: BaseDecisionTree,
        n_trees: int,
        max_samples: int,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int = None,
    ):
        super().__init__(
            tree=tree,
            n_trees=n_trees,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
        )

    def fit(self, X: NDArray, y: NDArray) -> "RandomForestClassifier":
        self.n_classes_ = len(np.unique(y))
        return self

    def predict(self, X: NDArray) -> NDArray:
        self.check_is_fitted()
        probas = self.predict_proba(X)  # shape: (n_samples, n_classes)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: NDArray) -> NDArray:
        self.check_is_fitted()
        probas = np.array(
            [tree.predict_proba(X) for tree in self.trees_]
        )  # shape: (n_trees, n_samples, n_classes)
        return np.mean(probas, axis=0)


class RandomForestRegressor(RandomForestBase):
    def __init__(
        self,
        tree: BaseDecisionTree,
        n_trees: int,
        max_samples: int,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int = None,
    ):
        super().__init__(
            tree=tree,
            n_trees=n_trees,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
        )

    def fit(self, X: NDArray, y: NDArray) -> "RandomForestRegressor":
        return self

    def predict(self, X: NDArray) -> NDArray:
        self.check_is_fitted()
        predictions = np.array(
            [tree.predict(X) for tree in self.trees_]
        )  # shape: (n_trees, n_samples)
        return np.mean(predictions, axis=0)
