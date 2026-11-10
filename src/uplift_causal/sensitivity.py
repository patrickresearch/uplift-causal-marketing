import numpy as np


def risk_ratio(y, t):
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(bool)
    return y[t].mean() / y[~t].mean()


def e_value(rr):
    rr = max(rr, 1 / rr)
    return rr + np.sqrt(rr * (rr - 1))


def robustness_value(estimate, se, dof, q=1.0):
    f = q * abs(estimate / se) / np.sqrt(dof)
    return 0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2)
