import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid, cumulative_simpson
import math

from typing import Callable


def inthelog(T, C = 0.7, r = 0.05):
    """For constant r"""
    # return (1-C) * np.exp(-r * T)
    return (1-C) / np.exp(r * T)

def inthelog2(T, C = 0.7, r = 0.05):
    """For continuous r, coming soon"""
    # return (1-C) * np.exp(-r * T)
    return (1-C) / np.exp(r * T)



def solve_epsilon(theta_norm_T: float, alpha: float, A: float) -> float:
    """
    solving epsilon* for VaR
    """
    z = float(norm.ppf(alpha))
    b = float(theta_norm_T) + z
    disc = b * b - 2.0 * math.log(A)
    return max(b + math.sqrt(disc), b - math.sqrt(disc))


def solve_epsilon_avar_lel(theta_norm_T: float, alpha: float, A: float, lel: bool = False,
                            *, x0=0, step=1.0, max_expand=200000,
                            bracket_min=-1e6, bracket_max=1e6,
                            xtol=1e-6, rtol=1e-6, maxiter=20000) -> float:
    """solving epsilon* for AVaR and LEL"""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if A <= 0.0:
        raise ValueError("A must be > 0 for ln(A).")
    theta = float(theta_norm_T)
    z = float(norm.ppf(alpha))
    # lnA = math.log(A)

    # we use logcdf for stability
    if lel:
        def h(eps:float) -> float:
            return norm.ppf(alpha) - norm.ppf(A) - eps
    else:
        def h(eps: float) -> float:
            return eps * theta - math.log(A) + norm.logcdf(z - eps)

    # bracket search around x0
    a = max(bracket_min, x0 - step)
    b = min(bracket_max, x0 + step)
    fa = h(a)
    fb = h(b)

    k = 0
    while fa * fb > 0 and k < max_expand:
        step *= 2.0
        a = max(bracket_min, x0 - step)
        b = min(bracket_max, x0 + step)
        fa = h(a)
        fb = h(b)
        k += 1

    if fa * fb > 0:
        raise ValueError(
            "Try a different x0/step."
        )

    return brentq(h, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter)


def opt_strategy(eps_star: float, theta_norm_T: float, Cov, B_df: pd.DataFrame):
    """
    Computing the optimal strategy at each t, will eventually be expanded for continuous time variances
    """
    if theta_norm_T == 0:
        raise ZeroDivisionError("theta_norm_T is 0; cannot divide by ||Theta||_T.")

    B = B_df.to_numpy(dtype=float)
    C = np.asarray(Cov, dtype=float)

    n, d = B.shape
    if C.shape != (d, d):
        raise ValueError(f"Cov must have shape {(d,d)}, got {C.shape}.")

    # Solve Cov X = B^T  => X = Cov^{-1} B^T  (X is d x n)
    X = np.linalg.solve(C, B.T).T 

    out = (eps_star / theta_norm_T) * X
    return pd.DataFrame(out, index=B_df.index, columns=B_df.columns)

def b_stock(u_, b_, phi_, t):
    return u_ + b_*np.cos(phi_ * t)

def B_t(b, r):
    return b - r

def time_grid_df(time, n=1000, col="t"):
    if time == 0:
        t = 0
        return t
    t = np.linspace(0.0, float(time), int(n))
    return pd.DataFrame({col: t})

def theta_norm_from_df(
    B_df: pd.DataFrame,
    T,
    Cov: np.ndarray,
    *,
    # n = 10000,
    time_col: str | None = None,
) -> float:
    if T == 0:
        return 0

    # def time_grid_df(time, n=1000, col="t"):
    #     if time == 0:
    #         t = 0
    #         return t
    #     t = np.linspace(0.0, float(time), int(n))
    #     return pd.DataFrame({col: t})

    B = B_df.to_numpy(dtype=float)
    n, d = B.shape
    time_df = time_grid_df(T, n=n)
    if isinstance(time_df, pd.DataFrame):
        col = time_col if time_col is not None else time_df.columns[0]
        t = time_df[col].to_numpy(dtype=float)
    elif isinstance(time_df, pd.Series):
        t = time_df.to_numpy(dtype=float)
    else:
        t = np.asarray(time_df, dtype=float)

    C = np.asarray(Cov, dtype=float)
    if C.shape != (d, d):
        raise ValueError(f"Cov must have shape {(d, d)}, got {C.shape}.")

    L = np.linalg.cholesky(C)  
    YT = np.linalg.solve(L, B.T)        
    integrand = np.sum(YT * YT, axis=0)   

    integral = np.trapezoid(integrand, t)
    return math.sqrt(float(integral))


def prep_r_col(df):
    df["r_w"] = 1 -  df["b_1"] - df["b_2"] - df["b_3"]
    return df

def split_3dfs_into_4dfs_var_avar_lel(df_var: pd.DataFrame,
                                      df_avar: pd.DataFrame,
                                      df_lel: pd.DataFrame,
                                      *,
                                      col_names=("var", "avar", "lel")):
    if not (df_var.index.equals(df_avar.index) and df_var.index.equals(df_lel.index)):
        raise ValueError("Input DataFrames must have the same index (same rows/order).")

    def make_k(k: int) -> pd.DataFrame:
        out = pd.concat(
            [
                df_var.iloc[:, k].rename(col_names[0]),
                df_avar.iloc[:, k].rename(col_names[1]),
                df_lel.iloc[:, k].rename(col_names[2]),
            ],
            axis=1,
        )
        return out

    return make_k(0), make_k(1), make_k(2), make_k(3)


def expected_wealth(eps, X0, r, T, B_df, covariance):
    if T == 0:
        theta = 0
        return X0 * np.exp(r * T) * np.exp(eps * theta)      
    # grid = time_grid_df(T, n=10000)
    theta = theta_norm_from_df(B_df, T, covariance)
    return X0 * np.exp(r * T) * np.exp(eps * theta)


def plot_theta(df, covariance, time = 8, n = 10000, time_dep: bool = False):
    t = np.linspace(0.0, time, n)
    theta_plot = [theta_norm_from_df(df, ti, covariance) for ti in t]

    plt.figure(figsize=(10, 5))
    plt.plot(t, theta_plot)
    plt.axhline(0.0, linewidth=1.0, alpha=0.6)
    plt.xlabel("time in years")
    plt.ylabel("Theta")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_helper(df: pd.DataFrame, T: float = 8.0, *, xlabel="time in years", ylabel="Portfolio weights", title=None):
    n = len(df)
    t = np.linspace(0.0, float(T), n)

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df.columns:
        ax.plot(t, df[col].to_numpy(dtype=float), label=str(col))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()


#######
## time variable
#######

def vol_matrix_t(
    t: float | np.ndarray,
    vt_func: Callable[[float | np.ndarray], np.ndarray],
    rhot: np.ndarray,
) -> np.ndarray:

    rhot = np.asarray(rhot, dtype=float)
    if np.ndim(t) == 0:
        V = np.asarray(vt_func(float(t)), dtype=float)
        return V @ rhot @ V
    return [vol_matrix_t(float(ti), vt_func, rhot) for ti in np.asarray(t)]



def theta_norm_td(B_df, T, vt_func, rhot):

    if T == 0:
        return 0.0
    B = B_df.to_numpy(dtype=float)
    n, d = B.shape
    t = np.linspace(0.0, T, n)
    rhot = np.asarray(rhot, dtype=float)

    integrand = np.empty(n, dtype=float)
    for i in range(n):
        V_i = np.asarray(vt_func(t[i]), dtype=float)
        Sigma_i = V_i @ rhot @ V_i
        b_i = B[i]
        integrand[i] = b_i @ np.linalg.solve(Sigma_i, b_i)

    cum = cumulative_trapezoid(integrand, t, initial=0)
    # cum = cumulative_simpson(integrand, x = t, initial=0)

    return t, np.sqrt(cum)


def opt_strategy_td(
    eps_star: float,
    theta_norm_T: float,
    vt_func: Callable[[float], np.ndarray],
    rhot: np.ndarray,
    B_df: pd.DataFrame,
    T: float,
) -> pd.DataFrame:
    
    if theta_norm_T == 0:
        raise ZeroDivisionError("theta_norm_T is 0; cannot divide by ‖Θ‖_T.")

    B = B_df.to_numpy(dtype=float)
    n, d = B.shape
    t = np.linspace(0.0, float(T), n)
    rhot = np.asarray(rhot, dtype=float)
    scale = eps_star / theta_norm_T

    out = np.empty_like(B)
    for i in range(n):
        V_i = np.asarray(vt_func(t[i]), dtype=float)
        Sigma_i = V_i @ rhot @ V_i
        out[i] = scale * np.linalg.solve(Sigma_i, B[i])

    return pd.DataFrame(out, index=B_df.index, columns=B_df.columns)


def expected_wealth_td(eps, X0, r, B_df, T, vt_func, rhot):
    t, theta = theta_norm_td(B_df, T, vt_func, rhot)
    return X0 * np.exp(r * t) * np.exp(eps * theta)