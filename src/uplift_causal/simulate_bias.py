import numpy as np


def true_propensity(df, weights, strength=1.0, clip=0.05):
    z = sum(w * (df[c] - df[c].mean()) / df[c].std() for c, w in weights.items())
    return np.clip(1 / (1 + np.exp(-strength * z.to_numpy())), clip, 1 - clip)


def nonlinear_propensity(df, weights, strength=1.0, clip=0.05):
    a, b = (w * (df[c] - df[c].mean()) / df[c].std() for c, w in list(weights.items())[:2])
    z = (a**2 + b**2 - 2).to_numpy()
    return np.clip(1 / (1 + np.exp(-strength * z)), clip, 1 - clip)


def simulate_bias(df, weights, strength=1.0, clip=0.05, seed=0, propensity=true_propensity):
    e = propensity(df, weights, strength, clip)
    t = df.t.to_numpy()
    p = t.mean()
    m = min(p, 1 - p)
    keep = np.where(t == 1, e * m / p, (1 - e) * m / (1 - p))
    mask = np.random.default_rng(seed).random(len(df)) < keep
    return df[mask].reset_index(drop=True)
