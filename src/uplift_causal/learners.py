from econml.dml import CausalForestDML
from econml.metalearners import TLearner, XLearner
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyClassifier


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


def fit_predict(X_train, t_train, y_train, X_test, seed=0, names=None):
    models = {k: v for k, v in learners(seed).items() if names is None or k in names}
    for m in models.values():
        m.fit(y_train, t_train, X=X_train)
    return models, {k: m.effect(X_test) for k, m in models.items()}


def baseline(t_learner, X):
    return t_learner.models[0].predict(X)
