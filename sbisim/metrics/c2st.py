# implementation from github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py

from typing import Optional

import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> np.ndarray:
    """Classifier-based 2-sample test returning accuracy

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples x and y.

    Args:
        x: Sample 1
        y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        scoring: Scoring metric for cross-validation
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    if z_score:
        X_mean = np.mean(x, axis=0)
        X_std = np.std(x, axis=0)
        X = (x - X_mean) / X_std
        Y = (y - X_mean) / X_std
    else:
        X = x
        Y = y

    if noise_scale is not None:
        X += noise_scale * np.random.randn(*X.shape)
        Y += noise_scale * np.random.randn(*Y.shape)

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return scores