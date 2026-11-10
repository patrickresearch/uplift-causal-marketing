import numpy as np
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

THREADS = None


def lgbm(cls=LGBMRegressor, seed=0, **kw):
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=THREADS,
        verbose=-1,
    )
    return cls(**params | kw)


def learners(seed=0):
    return {
        "T-Learner": TLearner(models=lgbm(seed=seed)),
        "X-Learner": XLearner(models=lgbm(seed=seed), propensity_model=DummyClassifier(strategy="prior")),
        "Causal Forest": CausalForestDML(
            model_y=lgbm(seed=seed),
            model_t=DummyClassifier(strategy="prior"),
            discrete_treatment=True,
            n_estimators=500,
            min_samples_leaf=50,
            cv=3,
            random_state=seed,
        ),
    }


def extra_learners(seed=0):
    return {
        "S-Learner": SLearner(overall_model=lgbm(seed=seed)),
        "DR-Learner": DRLearner(
            model_propensity=DummyClassifier(strategy="prior"),
            model_regression=lgbm(seed=seed),
            model_final=lgbm(seed=seed),
            cv=3,
            random_state=seed,
        ),
    }


def fit_predict(X_train, t_train, y_train, X_test, seed=0, names=None):
    available = learners(seed) | extra_learners(seed)
    names = list(learners(seed)) if names is None else names
    models = {k: available[k] for k in names}
    for m in models.values():
        m.fit(y_train, t_train, X=X_train)
    return models, {k: m.effect(X_test) for k, m in models.items()}


def baseline(t_learner, X):
    return np.clip(t_learner.models[0].predict(X), 0, 1)


def _fold(X, t, y, train, test, seed, names, threads):
    global THREADS
    THREADS = threads
    return test, fit_predict(X.iloc[train], t.iloc[train], y.iloc[train], X.iloc[test], seed, names)[1]


def cross_fit(X, t, y, names, n_splits=5, repeats=5, seed=0, n_jobs=-1):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed)
    folds = Parallel(n_jobs=n_jobs)(
        delayed(_fold)(X, t, y, train, test, seed + i, names, None if n_jobs == 1 else 1)
        for i, (train, test) in enumerate(cv.split(X, t * 2 + y))
    )
    out = {k: np.empty((repeats, len(X))) for k in names}
    for i, (test, tau) in enumerate(folds):
        for k in names:
            out[k][i // n_splits, test] = np.ravel(tau[k])
    return out
