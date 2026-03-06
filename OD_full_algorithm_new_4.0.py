import numpy as np
from numpy.random import default_rng
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.special import comb, factorial, gammaln
from itertools import product
from scipy.stats import poisson
from numpy.random import gamma, multivariate_normal, uniform, randint, multinomial
import time
import warnings
from properscoring import crps_ensemble
from multiprocessing import Pool
import ipyparallel as ipp
warnings.filterwarnings('ignore')
import math
from numba import njit
from math import lgamma
import multiprocessing as mp


# ------- Update phi -------

def compute_G(Phi, Psi):
    return Phi @ Psi.T

def compute_lambda(G, row_slices, eps=1e-12):
    lam = np.zeros_like(G, dtype=float)
    for sl in row_slices:
        Z = G[sl, :]
        Zmax = Z.max(axis=0, keepdims=True)
        E = np.exp(Z - Zmax)
        lam[sl, :] = E / (E.sum(axis=0, keepdims=True) + eps)
    return lam


def check_row_margins(Y, u, row_slices):
    for i, sl in enumerate(row_slices):
        rs = Y[:, sl].sum(axis=1)
        if not np.all(rs == u[:, i]):
            diff = rs - u[:, i]
            return False, i, diff
    return True, None, None


_row_cache = {}
def _get_row_start_len(row_slices):
    key = tuple(len(sl) for sl in row_slices)
    if key in _row_cache:
        return _row_cache[key]

    row_start = np.array([int(sl[0]) for sl in row_slices], dtype=np.int64)
    row_len   = np.array([int(len(sl)) for sl in row_slices], dtype=np.int64)
    _row_cache[key] = (row_start, row_len)
    return row_start, row_len

@njit
def compute_lambda_nb(G, row_start, row_len):
    """
    Numba version of block-wise softmax over rows of G.
    """
    M, N = G.shape
    lam = np.zeros((M, N), dtype=np.float64)

    n_blocks = row_start.shape[0]

    for b in range(n_blocks):
        start = row_start[b]
        K_i   = row_len[b]
        end   = start + K_i

        # softmax for each column
        for n in range(N):
            maxZ = -1e300
            for k in range(start, end):
                val = G[k, n]
                if val > maxZ:
                    maxZ = val

            sumE = 0.0
            for k in range(start, end):
                e = math.exp(G[k, n] - maxZ)
                lam[k, n] = e
                sumE += e

            inv_sumE = 1.0 / sumE
            for k in range(start, end):
                lam[k, n] *= inv_sumE

    return lam

@njit
def loglikelihood_nb(Phi, Psi, Y, row_start, row_len):
    """
    Log-likelihood of Y given Phi, Psi.
    Y: shape (N, M)
    """
    M, D = Phi.shape
    N, D2 = Psi.shape
    assert D == D2

    # 1) G = Phi @ Psi^T
    G = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        for n in range(N):
            s = 0.0
            for d in range(D):
                s += Phi[m, d] * Psi[n, d]
            G[m, n] = s

    # 2) lam = softmax blocks
    lam = compute_lambda_nb(G, row_start, row_len)

    # 3) loglik = sum_{n,m} Y[n,m] * log lam[m,n]
    eps = 1e-12
    total = 0.0
    for n in range(N):
        for m in range(M):
            y_nm = Y[n, m]
            if y_nm > 0:
                p = lam[m, n]
                if p < eps:
                    p = eps
                total += y_nm * math.log(p)
    return total

def loglikelihood(Phi, Psi, Y, u, row_slices):

    row_start, row_len = _get_row_start_len(row_slices)

    Phi = np.asarray(Phi, np.float64)
    Psi = np.asarray(Psi, np.float64)
    Y   = np.asarray(Y,   np.int64)

    return float(loglikelihood_nb(Phi, Psi, Y, row_start, row_len))

def logprior_phi(Phi, sigma0):
    return -0.5 * (Phi**2).sum() / (sigma0**2)

def logposterior_phi(Phi, Psi, Y, u, row_slices, sigma0):
    return loglikelihood(Phi, Psi, Y, u, row_slices) + logprior_phi(Phi, sigma0)

def residuals_matrix(Phi, Psi, Y, u, row_slices):
    G = compute_G(Phi, Psi)
    lam = compute_lambda(G, row_slices)
    R_list = [Y[:, sl] - u[:, i, None] * lam[sl, :].T for i, sl in enumerate(row_slices)]
    return R_list, lam

def grad_phi_blocks(Phi, Psi, R_list, sigma0, row_slices):
    """
    Gradient wrt Phi when loglik uses softmax(G).
    """
    grad_full = np.zeros_like(Phi)
    grad_blocks = []
    for i, sl in enumerate(row_slices):
        R_i = R_list[i]              # (N, K_i)
        G_i = R_i.T @ Psi            # (K_i, D)
        G_i -= Phi[sl, :] / (sigma0**2)
        grad_blocks.append(G_i)
        grad_full[sl, :] = G_i
    return grad_blocks, grad_full

def _project_grad_blocks_zero_sum(grad_blocks):
    return [g - g.mean(axis=0, keepdims=True) for g in grad_blocks]

def precond_M_block(lam_block, u_i, sigma0, Psi,
                    mode="k_scaled",
                    M_inflate=1.0,
                    eps=1e-12,
                    m_min=1e-8,
                    m_max=None):

    K_i, N = lam_block.shape
    N2, D  = Psi.shape
    assert N == N2

    prior_prec = 1.0 / (sigma0**2)

    lam = lam_block.clip(1e-9, 1-1e-9)
    w_kn = (u_i[None, :] * lam * (1.0 - lam))   # (K_i, N)

    if mode == "k_scaled":
        Psi_sq = Psi**2                          # (N, D)
        H = w_kn @ Psi_sq                        # (K_i, D)
    elif mode == "simple":
        H_row = w_kn.sum(axis=1)                 # (K_i,)
        H = H_row[:, None]                       # (K_i, D)
    elif mode == "psi_scaled":
        H_row = w_kn.sum(axis=1)                 # (K_i,)
        psi_colsumsq = np.maximum(np.sum(Psi**2, axis=0), eps)  # (D,)
        H = H_row[:, None] * psi_colsumsq[None, :]
    else:
        raise ValueError("mode must be 'k_scaled' | 'simple' | 'psi_scaled'")

    H_block = np.maximum(H + prior_prec, eps)
    M_block = M_inflate / H_block

    if m_min is not None:
        M_block = np.maximum(M_block, m_min)
    if m_max is not None:
        M_block = np.minimum(M_block, m_max)

    return M_block, H_block


import numpy as np

def get_ref_idx(row_slices):
    """
    Reference-cell row indices: for each origin block i, use sl[-1] as reference row.
    """
    return np.array([int(sl[-1]) for sl in row_slices], dtype=int)

# - pMALA -
def pmala_update_phi(
    Phi, Psi, Y, u, sigma0, row_slices, h_phi, rng,
    eps=1e-12,
    M_mode="k_scaled",
    M_inflate=1.0,
    m_min=1e-8,
    m_max=None,
    h_min=1e-4,
    h_max=5.0,
):
    """
    Reference-cell (last category fixed) block-wise pMALA for Phi.

    For each origin block i with indices sl (length K_i):
      - free rows: sl_free = sl[:-1]
      - reference row: sl_ref = sl[-1], fixed Phi[sl_ref,:] == 0 always
    This implies G_ref = 0 and softmax over [G_free; 0] matches the reference-cell parametrization.

    Returns same outputs as your original pmala_update_phi.
    """
    M, D = Phi.shape
    N = Psi.shape[0]
    n_blocks = len(row_slices)

    # --- enforce reference rows to be exactly zero (project onto constrained space) ---
    ref_idx = get_ref_idx(row_slices)
    Phi0 = Phi.copy()
    Phi0[ref_idx, :] = 0.0

    # --- global lambda and current log-posterior (constant shift from fixed ref is irrelevant) ---
    G = compute_G(Phi0, Psi)
    lam = compute_lambda(G, row_slices, eps=eps)
    current_logpost = logposterior_phi(Phi0, Psi, Y, u, row_slices, sigma0)

    Phi_new = Phi0.copy()

    block_accept_cum = np.zeros(n_blocks, dtype=int)
    block_trials_cum = np.zeros(n_blocks, dtype=int)
    block_accepted_now = np.zeros(n_blocks, dtype=int)

    order = np.arange(n_blocks)
    rng.shuffle(order)

    for idx in order:
        i = int(idx)
        sl = row_slices[i]
        K_i = len(sl)

        block_trials_cum[i] += 1

        # step size
        if np.ndim(h_phi) == 0:
            h_i = float(h_phi)
        else:
            if len(h_phi) != n_blocks:
                raise ValueError(f"h_phi length {len(h_phi)} != n_blocks {n_blocks}")
            h_i = float(h_phi[i])

        sl_free = sl[:-1]
        sl_ref = int(sl[-1])

        # K_i == 1: only reference category exists -> probability is identically 1, nothing to update
        if K_i <= 1:
            Phi_new[sl_ref, :] = 0.0
            continue

        # --- old state (free part only) ---
        Phi_free_old = Phi_new[sl_free, :].copy()      # (K_i-1, D)
        lam_block_old = lam[sl, :].copy()              # (K_i, N)
        lam_free_old = lam_block_old[:-1, :]           # (K_i-1, N)

        Y_block = Y[:, sl]                             # (N, K_i)
        Y_free = Y_block[:, :-1]                       # (N, K_i-1)
        u_i = np.asarray(u[:, i], dtype=float)         # (N,)

        # --- gradient at old state (free only) ---
        R_free_old = Y_free - u_i[:, None] * lam_free_old.T   # (N, K_i-1)
        grad_free_old = R_free_old.T @ Psi - Phi_free_old / (sigma0 ** 2)  # (K_i-1, D)

        # --- diagonal preconditioner at old (free only) ---
        if M_mode == "none":
            M_free = np.ones_like(Phi_free_old)
        else:
            M_free, _ = precond_M_block(
                lam_block=lam_free_old,
                u_i=u_i,
                sigma0=sigma0,
                Psi=Psi,
                mode=M_mode,
                M_inflate=M_inflate,
                eps=eps,
                m_min=m_min,
                m_max=m_max,
            )

        # --- forward proposal in free space ---
        mean_fwd = Phi_free_old + 0.5 * h_i * (M_free * grad_free_old)
        sd_fwd = np.sqrt(h_i) * np.sqrt(np.maximum(M_free, eps))
        Phi_free_prop = mean_fwd + sd_fwd * rng.normal(size=grad_free_old.shape)

        # --- proposed lambda (build logits as [G_free; 0]) ---
        G_free_prop = Phi_free_prop @ Psi.T                     # (K_i-1, N)
        G_block_prop = np.vstack([G_free_prop, np.zeros((1, N))])  # (K_i, N), ref logits == 0

        G_shift = G_block_prop - G_block_prop.max(axis=0, keepdims=True)
        E = np.exp(G_shift)
        lam_block_prop = E / (E.sum(axis=0, keepdims=True) + eps)   # (K_i, N)
        lam_free_prop = lam_block_prop[:-1, :]

        # --- log-likelihood contributions (FULL block incl. reference column!) ---
        Y_block_T = Y_block.T  # (K_i, N)
        loglik_old_block = float(np.sum(Y_block_T * np.log(lam_block_old + eps)))
        loglik_new_block = float(np.sum(Y_block_T * np.log(lam_block_prop + eps)))

        # --- prior contributions (free only; ref row is fixed at 0 so it cancels) ---
        prior_old_block = -0.5 / (sigma0 ** 2) * float(np.sum(Phi_free_old ** 2))
        prior_new_block = -0.5 / (sigma0 ** 2) * float(np.sum(Phi_free_prop ** 2))

        # --- gradient at proposed state (free only) ---
        R_free_new = Y_free - u_i[:, None] * lam_free_prop.T
        grad_free_new = R_free_new.T @ Psi - Phi_free_prop / (sigma0 ** 2)

        # --- preconditioner at proposed state (free only) ---
        if M_mode == "none":
            M_free_p = np.ones_like(Phi_free_prop)
        else:
            M_free_p, _ = precond_M_block(
                lam_block=lam_free_prop,
                u_i=u_i,
                sigma0=sigma0,
                Psi=Psi,
                mode=M_mode,
                M_inflate=M_inflate,
                eps=eps,
                m_min=m_min,
                m_max=m_max,
            )

        # --- proposal log densities in FREE space ---
        def _log_q_free(from_block, grad_from, to_block, M_blk, h):
            mu = from_block + 0.5 * h * (M_blk * grad_from)
            sd = np.sqrt(h) * np.sqrt(np.maximum(M_blk, eps))
            z = (to_block - mu) / sd
            return -0.5 * (
                np.sum(z * z) + np.sum(2.0 * np.log(sd)) + to_block.size * np.log(2.0 * np.pi)
            )

        logq_fwd = _log_q_free(Phi_free_old, grad_free_old, Phi_free_prop, M_free, h_i)
        logq_bwd = _log_q_free(Phi_free_prop, grad_free_new, Phi_free_old, M_free_p, h_i)

        # --- MH ratio ---
        delta_logpost_block = (loglik_new_block + prior_new_block) - (loglik_old_block + prior_old_block)
        logr = delta_logpost_block + (logq_bwd - logq_fwd)

        if np.log(rng.uniform()) < logr:
            # accept: write free rows, enforce ref row zero, update lambda cache for this block
            Phi_new[sl_free, :] = Phi_free_prop
            Phi_new[sl_ref, :] = 0.0
            lam[sl, :] = lam_block_prop

            current_logpost += delta_logpost_block
            block_accept_cum[i] += 1
            block_accepted_now[i] = 1
        else:
            # reject: keep old free rows, still enforce ref row zero
            Phi_new[sl_ref, :] = 0.0

    # final safety
    Phi_new[ref_idx, :] = 0.0

    return Phi_new, block_accept_cum, block_trials_cum, block_accepted_now, current_logpost



# ------ Update psi ------
def periodic_kernel(x1, x2, length_scale, period):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    sine_squared = np.sin(np.pi * np.abs(x1 - x2.T) / period) ** 2
    K = np.exp(-2 * sine_squared / length_scale**2)
    return K

def exponential_kernel(x, x_prime, l_e):
    x1 = np.atleast_2d(x)
    x2 = np.atleast_2d(x_prime)
    return np.exp(-np.power(x1 - x2.T,2) / (2*(l_e**2)))

def combined_kernel(x, x_prime, p, l_p, l_e, w):
    return (1-w)*periodic_kernel(x, x_prime, l_p, p) + w*exponential_kernel(x, x_prime, l_e)

def elliptical_slice_sampling_Psi(d, g_d, choll, N, Phi_s, Psi_s, Y, u, row_slices, rng):
    """
    ESS update for Psi[:, d], using the passed-in numpy Generator rng.
    """
    # draw nu ~ N(0, K)
    nu = choll @ rng.normal(0.0, 1.0, N)

    # slice threshold: log y = loglik(current) + log u
    logy = loglikelihood(Phi_s, Psi_s, Y, u, row_slices) + np.log(rng.random())

    theta     = rng.uniform(0.0, 2.0 * np.pi)
    theta_min = theta - 2.0 * np.pi
    theta_max = theta

    Psi_temp = Psi_s.copy()

    while True:
        g_d_prime = g_d * np.cos(theta) + nu * np.sin(theta)
        Psi_temp[:, d] = g_d_prime

        if loglikelihood(Phi_s, Psi_temp, Y, u, row_slices) > logy:
            return g_d_prime

        # shrink bracket
        if theta <= 0.0:
            theta_min = theta
        else:
            theta_max = theta

        theta = rng.uniform(theta_min, theta_max)

# ------ Update Y ------

def compute_W_from_p(p, eps=1e-12):
    p = np.asarray(p, dtype=float)
    S = p.shape[0]
    assert p.ndim == 2 and p.shape[1] == S, "p must be square (S×S)."

    W = np.zeros((S, S), dtype=float)
    for i in range(S):
        tail = np.cumsum(p[i, ::-1])[::-1]
        for j in range(i + 1, S):
            denom = tail[j + 1] if (j + 1 < S) else 0.0
            W[i, j] = p[i, j] / denom if denom > eps else 0.0

    return W

@njit
def compute_W_from_p(p, eps=1e-12):
    S = p.shape[0]
    W = np.zeros((S, S))

    for i in range(S - 1):
        tail = np.zeros(S)
        acc = 0.0
        for idx in range(S - 1, -1, -1):
            acc += p[i, idx]
            tail[idx] = acc

        for j in range(i + 1, S):
            if j + 1 < S:
                denom = tail[j + 1]
            else:
                denom = 0.0

            if denom > eps:
                W[i, j] = p[i, j] / denom
            else:
                W[i, j] = 0.0

    return W


from numba import njit
import math

@njit
def compute_W_from_pvec_nb(p_vec, row_start, row_len, S, eps=1e-12, min_w=1e-15):
    """
    Build W (S×S) from p_vec (length M), without constructing p_mat.

    row_start[i], row_len[i] describe the slice in p_vec corresponding to origin i,
    i.e. destinations j=i+1..S-1 in order.

    W[i,j] = p[i,j] / sum_{k=j+1..S-1} p[i,k]
    and W[i,S-1] = 0 by definition (denom=0).
    """
    W = np.zeros((S, S), dtype=np.float64)

    for i in range(S - 1):
        start = row_start[i]
        K = row_len[i]           # should be S-i-1
        if K <= 1:
            # only last column exists, denom=0 -> W stays 0
            continue

        # suffix sums of p_row
        # p_row[t] corresponds to dest j = i+1+t, for t=0..K-1
        # suffix[t] = sum_{r=t..K-1} p_row[r]
        suffix = np.zeros(K, dtype=np.float64)
        acc = 0.0
        for t in range(K - 1, -1, -1):
            acc += p_vec[start + t]
            suffix[t] = acc

        # fill W[i, j] for j=i+1..S-2 (t=0..K-2); last col stays 0
        for t in range(K - 1):
            denom = suffix[t + 1]
            if denom > eps:
                val = p_vec[start + t] / denom
                # keep strictly positive weights for Fisher sampler stability
                if val < min_w:
                    val = min_w
                W[i, i + 1 + t] = val
            else:
                W[i, i + 1 + t] = min_w

    return W

def compute_W_from_pvec(p_vec, row_start, row_len, S, eps=1e-12, min_w=1e-15):
    p_vec = np.asarray(p_vec, dtype=np.float64)
    row_start = np.asarray(row_start, dtype=np.int64)
    row_len = np.asarray(row_len, dtype=np.int64)
    return compute_W_from_pvec_nb(p_vec, row_start, row_len, int(S), eps, min_w)

def tilt_pvec_blockwise(p_vec, row_start, row_len, alpha=0.7, eps=1e-12):
    p_vec = np.asarray(p_vec, dtype=np.float64)
    out = p_vec.copy()
    n_blocks = len(row_start)

    for b in range(n_blocks):
        s = int(row_start[b])
        K = int(row_len[b])
        sl = slice(s, s + K)

        tmp = np.power(out[sl] + eps, alpha)
        z = tmp.sum()
        if z > 0:
            out[sl] = tmp / z
        else:
            out[sl] = 1.0 / K

    return out


@njit
def _log_binom_nb(m, x):
    """
    Numba version of log C(m, x).
    """
    if x < 0 or x > m:
        return -np.inf
    return lgamma(m + 1.0) - lgamma(x + 1.0) - lgamma(m - x + 1.0)

def _log_coeff_row(m: int, w: float, n: int) -> np.ndarray:
    """
    Return an array log_c[0..min(m,n)] where
      log_c[x] = log C(m, x) + x * log w
    Vectorized implementation (no Python loop over x).
    """
    if w <= 0.0:
        raise ValueError("All weights/odds must be > 0 for Fisher's distribution.")

    xmax = min(m, n)
    # xs = 0,1,...,xmax
    xs = np.arange(xmax + 1, dtype=np.int32)

    # log C(m, x) = lgamma(m+1) - lgamma(x+1) - lgamma(m-x+1)
    log_binoms = (
        gammaln(m + 1.0)
        - gammaln(xs + 1.0)
        - gammaln(m - xs + 1.0)
    )

    logw = np.log(w)
    logc = log_binoms + xs * logw
    return logc.astype(np.float64)


@njit
def sample_fisher_multivariate_logspace_nb(m, w, n):
    """
    Numba core: exact sampler for multivariate Fisher's noncentral
    hypergeometric distribution, using DP in log-space.

    m: int64 array (c,)
    w: float64 array (c,)
    n: int
    """
    c = m.shape[0]

    total_m = 0
    for i in range(c):
        if m[i] < 0:
            raise ValueError("Capacities m must be nonnegative.")
        total_m += m[i]
    if n < 0 or n > total_m:
        raise ValueError("Invalid n: must be between 0 and sum(m).")
    if n == 0:
        return np.zeros(c, dtype=np.int64)

    for i in range(c):
        if w[i] <= 0.0:
            raise ValueError("Weights w must be strictly positive.")

    # 1) log_coeffs[i, x] = log C(m_i, x) + x log w_i
    log_coeffs = np.full((c, n + 1), -np.inf)
    for i in range(c):
        mi = m[i]
        wi = w[i]
        xmax = mi if mi < n else n
        logw = np.log(wi)
        for x in range(xmax + 1):
            log_coeffs[i, x] = _log_binom_nb(mi, x) + x * logw

    # 2) DP: B[i, k] = log total weight allocating k items to colors i..c-1
    B = np.full((c + 1, n + 1), -np.inf)
    B[c, 0] = 0.0

    for i in range(c - 1, -1, -1):
        Bi = B[i]
        Bj = B[i + 1]

        # reset Bi
        for k in range(n + 1):
            Bi[k] = -np.inf

        xmax = n
        for x in range(n, -1, -1):
            if log_coeffs[i, x] > -np.inf:
                xmax = x
                break

        for x in range(xmax + 1):
            log_ci_x = log_coeffs[i, x]
            if log_ci_x == -np.inf:
                continue

            # Bi[k] = logaddexp(Bi[k], log_ci_x + Bj[k-x]), k >= x
            for k in range(x, n + 1):
                val = log_ci_x + Bj[k - x]
                a = Bi[k]
                if a == -np.inf:
                    Bi[k] = val
                else:
                    if a > val:
                        Bi[k] = a + np.log(1.0 + np.exp(val - a))
                    else:
                        Bi[k] = val + np.log(1.0 + np.exp(a - val))

    # 3) forward sampling with unnormalized weights + CDF + Uniform
    x = np.zeros(c, dtype=np.int64)
    krem = n

    tmpL = np.empty(n + 1, dtype=np.float64)
    weights = np.empty(n + 1, dtype=np.float64)
    cum = np.empty(n + 1, dtype=np.float64)

    for i in range(c):
        if krem == 0:
            break

        # xmax ≤ min(m[i], krem)
        xmax = m[i] if m[i] < krem else krem
        if xmax < 0:
            continue

        # L[x] = log_coeffs[i, x] + B[i+1, krem-x]
        maxL = -np.inf
        for xval in range(xmax + 1):
            val = log_coeffs[i, xval] + B[i + 1, krem - xval]
            tmpL[xval] = val
            if val > maxL:
                maxL = val

        tot = 0.0
        for xval in range(xmax + 1):
            wgt = np.exp(tmpL[xval] - maxL)
            weights[xval] = wgt
            tot += wgt

        # build CDF
        acc = 0.0
        for xval in range(xmax + 1):
            acc += weights[xval]
            cum[xval] = acc

        # sample
        u = np.random.random() * cum[xmax]
        draw = 0
        while draw < xmax and u > cum[draw]:
            draw += 1

        x[i] = draw
        krem -= draw

    # 4) repair if krem != 0
    if krem != 0:
        for i in range(c - 1, -1, -1):
            spare = m[i] - x[i]
            if spare < 0:
                spare = 0
            take = krem if krem < spare else spare
            x[i] += take
            krem -= take
            if krem == 0:
                break
        if krem != 0:
            raise RuntimeError("Failed to allocate all draws; check inputs.")

    return x

def sample_fisher_multivariate_logspace(m, w, n, rng=None):
    """
    Python wrapper that keeps the old signature but ignores rng
    and calls the numba-compiled core.
    """
    m = np.asarray(m, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    n = int(n)
    return sample_fisher_multivariate_logspace_nb(m, w, n)


@njit
def fisher_logZ_core_nb(m, w, n):
    """
    Numba core for log Z_n = log( [z^n] prod_i (1 + w_i z)^{m_i} ).
    This is the same DP as before, but fully in nopython mode.
    """
    c = m.shape[0]
    total_m = 0
    for i in range(c):
        total_m += m[i]
    if n < 0 or n > total_m:
        return -np.inf

    # log_coeffs[i, x] = log C(m_i, x) + x log w_i, x=0..min(m_i, n)
    log_coeffs = np.full((c, n + 1), -np.inf)
    for i in range(c):
        mi = m[i]
        wi = w[i]
        if mi < 0 or wi <= 0.0:
            return -np.inf
        xmax = mi if mi < n else n
        logw = np.log(wi)
        for x in range(xmax + 1):
            log_coeffs[i, x] = _log_binom_nb(mi, x) + x * logw

    # B[i, k] = log total weight of allocating k items to colors i..c-1
    B = np.full((c + 1, n + 1), -np.inf)
    B[c, 0] = 0.0

    for i in range(c - 1, -1, -1):
        Bi = B[i]
        Bj = B[i + 1]

        # reset Bi
        for k in range(n + 1):
            Bi[k] = -np.inf

        xmax = n
        for x in range(n, -1, -1):
            if log_coeffs[i, x] > -np.inf:
                xmax = x
                break

        for x in range(xmax + 1):
            log_ci_x = log_coeffs[i, x]
            if log_ci_x == -np.inf:
                continue

            # Bi[k] = logaddexp(Bi[k], log_ci_x + Bj[k-x]) for k >= x
            for k in range(x, n + 1):
                val = log_ci_x + Bj[k - x]
                a = Bi[k]
                if a == -np.inf:
                    Bi[k] = val
                else:
                    # manual logaddexp
                    if a > val:
                        Bi[k] = a + np.log(1.0 + np.exp(val - a))
                    else:
                        Bi[k] = val + np.log(1.0 + np.exp(a - val))

    return float(B[0, n])

def fisher_logZ_n(m, w, n):
    """
    Python wrapper around fisher_logZ_core_nb.
    Keeps the old interface: can pass list / np.array.
    """
    m = np.asarray(m, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    return float(fisher_logZ_core_nb(m, w, int(n)))

def sample_Y_once(A_col, B_row, W, rng=None, eps_w=1e-15, eps=1e-12):
    """
    One exact mFNCH draw of Y (S×S upper-triangular) AND its proposal log-density log q(Y)
    under the same column-by-column mFNCH construction.

    Returns
    -------
    Y : (S,S) int
    log_q : float
    """
    if rng is None:
        rng = np.random.default_rng()

    A_col = np.asarray(A_col, dtype=int)   # alightings
    B_row = np.asarray(B_row, dtype=int)   # boardings
    W     = np.asarray(W, dtype=float)

    S = W.shape[0]
    assert W.shape == (S, S)
    assert A_col.shape == (S,)
    assert B_row.shape == (S,)
    assert int(A_col.sum()) == int(B_row.sum()), "Total A != Total B"

    # IMPORTANT: your Fisher sampler requires strictly positive weights.
    # To keep proposal q consistent, we clip weights here and will use the clipped weights
    # in the log_q calculation too.
    W_use = np.where(np.isfinite(W) & (W > 0.0), W, eps_w)

    U_rem = B_row.copy()
    Y     = np.zeros((S, S), dtype=int)
    log_q = 0.0

    for j in range(0, S - 1):
        n_col = int(A_col[j])
        rows = np.arange(0, j)  # i < j

        if n_col == 0:
            continue
        if rows.size == 0:
            raise RuntimeError(f"[col {j}] no eligible rows (i<j). Expect A_col[0]==0, got {n_col}.")

        m_vec = U_rem[rows].astype(np.int64)      # capacities
        cap_before = int(m_vec.sum())
        if n_col > cap_before:
            raise RuntimeError(f"[col {j}] demand {n_col} > remaining cap {cap_before} over rows < {j}.")

        w_vec = W_use[rows, j].astype(np.float64)

        # draw x ~ Fisher(m_vec, w_vec; n_col)
        x = sample_fisher_multivariate_logspace(m=m_vec, w=w_vec, n=n_col, rng=rng)

        # fill and update remaining
        Y[rows, j] = x
        U_rem[rows] -= x

        # --- accumulate log q contribution for this column ---
        # log q_j(x) = sum_i [log C(m_i, x_i) + x_i log w_i] - logZ(m,w,n_col)
        # use vectorized gammaln for speed
        xs = x.astype(np.int64)
        log_num = (
            gammaln(m_vec + 1.0)
            - gammaln(xs + 1.0)
            - gammaln(m_vec - xs + 1.0)
            + xs * np.log(w_vec + eps)
        ).sum()

        logZ = fisher_logZ_n(m_vec, w_vec, n_col)
        log_q += (log_num - logZ)

        # checks (same as your original)
        if int(x.sum()) != n_col:
            raise AssertionError(f"[col {j}] sampled sum {int(x.sum())} != n_col {n_col}")
        if (U_rem[rows] < 0).any():
            bad = rows[U_rem[rows] < 0]
            raise AssertionError(f"[col {j}] negative remaining at rows {bad.tolist()}")

    # last column deterministic
    rem_last = int(U_rem[:S-1].sum())
    if rem_last != int(A_col[S-1]):
        raise RuntimeError(f"[last col] remaining sum {rem_last} != A_col[S-1] {int(A_col[S-1])}")
    if U_rem[S-1] != 0:
        raise RuntimeError(f"[last row] U_rem[S-1]={int(U_rem[S-1])} should be 0 (no j>i cells).")

    Y[:S-1, S-1] = U_rem[:S-1]

    # zero lower triangle + diagonal
    for i in range(S):
        Y[i, :i+1] = 0

    # final margin checks
    if not np.all(Y.sum(axis=1) == B_row):
        i_bad = np.where(Y.sum(axis=1) != B_row)[0]
        raise RuntimeError(f"Row-sum mismatch at rows {i_bad.tolist()}.")
    if not np.all(Y.sum(axis=0) == A_col):
        j_bad = np.where(Y.sum(axis=0) != A_col)[0]
        raise RuntimeError(f"Column-sum mismatch at cols {j_bad.tolist()}.")

    return Y, float(log_q)

# -- MH for Y --

# Multinomial: log π(Y)

@njit
def log_target_Y_from_pvec_nb(Y, p_vec, row_start, row_len, S, eps=1e-12):
    """
    log π(Y) = sum_{i<j} y_ij * log p_ij - lgamma(y_ij+1)
    where p_ij comes from p_vec via row_start/row_len mapping.
    """
    total = 0.0
    for i in range(S - 1):
        start = row_start[i]
        K = row_len[i]
        for t in range(K):
            j = i + 1 + t
            y = Y[i, j]
            if y > 0:
                p = p_vec[start + t]
                if p < eps:
                    p = eps
                total += y * math.log(p) - lgamma(y + 1.0)
    return total

def log_target_Y_from_pvec(Y, p_vec, row_start, row_len, S, eps=1e-12):
    Y = np.asarray(Y, dtype=np.int64)
    p_vec = np.asarray(p_vec, dtype=np.float64)
    row_start = np.asarray(row_start, dtype=np.int64)
    row_len = np.asarray(row_len, dtype=np.int64)
    return float(log_target_Y_from_pvec_nb(Y, p_vec, row_start, row_len, int(S), eps))

# mFNCH proposal: log q(Y)
@njit
def log_proposal_mfnch_nb(Y, U, V, W):
    """
    Numba version of log_proposal_mfnch.
    Y, U, V, W are assumed to be:
      - Y: int64 (S, S)
      - U, V: int64 (S,)
      - W: float64 (S, S)
    """
    S = Y.shape[0]

    # copy U into a working buffer
    U_rem = U.copy()
    log_q = 0.0

    for j in range(S - 1):
        n_col = V[j]
        if n_col == 0:
            continue

        # rows = 0..j-1
        if j == 0:
            # no eligible rows
            return -np.inf

        # sum y_col must equal n_col
        s = 0
        for idx in range(j):
            s += Y[idx, j]
        if s != n_col:
            return -np.inf

        # accumulate numerator log terms
        log_num = 0.0
        for idx in range(j):
            mi = U_rem[idx]
            yi = Y[idx, j]
            if yi < 0 or yi > mi:
                return -np.inf
            log_num += _log_binom_nb(mi, yi) + yi * np.log(W[idx, j])

        # denominator: logZ_n for this column
        # m_vec = U_rem[0:j], w_vec = W[0:j, j]
        m_vec = U_rem[:j]
        w_vec = W[:j, j]
        logZ = fisher_logZ_core_nb(m_vec, w_vec, n_col)

        log_q += (log_num - logZ)

        # update U_rem
        for idx in range(j):
            U_rem[idx] = U_rem[idx] - Y[idx, j]
            if U_rem[idx] < 0:
                return -np.inf

    # check last column consistency
    rem_last = 0
    for i in range(S - 1):
        rem_last += U_rem[i]
    if rem_last != V[S - 1]:
        return -np.inf
    if U_rem[S - 1] != 0:
        return -np.inf

    return float(log_q)

def log_proposal_mfnch(Y, U, V, W):
    """
    Python wrapper: same signature as before, but inside calls JITted core.
    """
    Y = np.asarray(Y, dtype=np.int64)
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    W = np.asarray(W, dtype=np.float64)
    return float(log_proposal_mfnch_nb(Y, U, V, W))

def mh_step_Y(current_Y, U, V, p_vec, row_start, row_len, S, W=None, rng=None):
    """
    One MH update for OD matrix Y using:
      - target log π(Y) computed from p_vec (no p_mat)
      - proposal q from mFNCH with odds W (computed from p_vec if not provided)
      - proposal draw returns log_q_prop directly (avoid recomputing log q for proposal)

    Returns: Y_new, accepted
    """
    if rng is None:
        rng = np.random.default_rng()

    current_Y = np.asarray(current_Y, dtype=np.int64)
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    p_vec = np.asarray(p_vec, dtype=np.float64)

    if W is None:
        W = compute_W_from_pvec(p_vec, row_start, row_len, S)

    # current logs (must be under CURRENT p_vec/W)
    log_pi_cur = log_target_Y_from_pvec(current_Y, p_vec, row_start, row_len, S)
    log_q_cur  = log_proposal_mfnch(current_Y, U, V, W)

    # proposal + its log_q (no extra DP for log_q_prop)
    Y_prop, log_q_prop = sample_Y_once(V, U, W, rng=rng)
    log_pi_prop = log_target_Y_from_pvec(Y_prop, p_vec, row_start, row_len, S)

    log_alpha = (log_pi_prop - log_pi_cur) + (log_q_cur - log_q_prop)

    if np.log(rng.random()) < min(0.0, log_alpha):
        return Y_prop, True
    else:
        return current_Y, False

# --------- MCMC ---------

def upper_index(S):
    pairs, row_slices, idx = [], [], 0
    for i in range(S - 1):
        K_i = S - i - 1
        idxs = np.arange(idx, idx + K_i, dtype=int)
        row_slices.append(idxs)
        for j in range(i + 1, S):
            pairs.append((i, j))
        idx += K_i
    return pairs, row_slices, idx

def get_upper_right(matrix, pairs=None):
    S = matrix.shape[0]
    if pairs is None:
        pairs, _, _ = upper_index(S)
    return np.array([matrix[i, j] for (i, j) in pairs], dtype=matrix.dtype)

def run_mcmc(
    B_mat,
    A_mat,
    Y_mat_true=None,
    epoch=1000,
    seed=2025,
    D=2,
    l_e=60.0,
    l_p=1.0,
    period=24 * 60.0,
    w=1.0,
    h_phi=0.1,             
    sigma0=1.0,
    burnin=200,
    thin=100,
    print_every=100,
    gamma_tilt=0.00,
    alpha_tilt=0.7,
    eps_tilt=1e-12,
    row_margin_debug=True,
    row_margin_check_every=100,
):
    """
    MCMC routine (single chain):
      - Y: mFNCH proposal + MH (uses p_vec; W from p_vec; proposal returns log_q_prop)
      - Phi: pMALA
      - Psi: ESS
      - P: softmax(G) computed each iteration (NOT stored)

    Storage:
      - Save y/Phi/Psi only for kept indices (after burn-in, thinned)
      - Save loglikelihood for ALL iterations (epoch)
      - Do NOT save P
    """
    import numpy as np
    import time
    import inspect
    from numpy.random import default_rng
    from scipy.special import gammaln

    assert 0 <= burnin < epoch
    assert thin >= 1

    # Important for numba routines that use np.random internally
    np.random.seed(seed)
    rng = default_rng(seed)

    B_mat = np.asarray(B_mat, dtype=np.int64)
    A_mat = np.asarray(A_mat, dtype=np.int64)
    bus_num, S = B_mat.shape
    N = bus_num

    # upper-triangular indexing
    pairs, row_slices, M = upper_index(S)
    print(f"[run_mcmc] N_bus={bus_num}, S={S}, M={M}, N={N}")

    n_blocks = len(row_slices)
    h_phi_per_block = np.full(n_blocks, h_phi, dtype=float)  
    target_accept = 0.45
    adapt_gamma = 0.05
    h_min = 1e-4
    h_max = 5.0

    accept_rate_ema = np.full(n_blocks, target_accept, dtype=float) 
    ema_decay = 0.95

    # row_start/row_len mapping for p_vec (length M)
    # row_slices length is S-1, each corresponds to origin i
    row_start_y = np.array([int(sl[0]) for sl in row_slices], dtype=np.int64)
    row_len_y   = np.array([int(len(sl)) for sl in row_slices], dtype=np.int64)

    # keep indices (thin AFTER burn-in)
    keep_idx = np.arange(burnin, epoch, thin)
    n_keep = len(keep_idx)
    print(f"  Will keep {n_keep} draws (burnin={burnin}, thin={thin}).")

    # Ground truth for loglikelihood (optional)
    has_truth = Y_mat_true is not None
    if has_truth:
        Y_mat_true = np.asarray(Y_mat_true, dtype=np.int64)
        assert Y_mat_true.shape == (bus_num, S, S)

        OD_vec = np.array(
            [get_upper_right(Y_mat_true[i]) for i in range(bus_num)],
            dtype=np.int64
        ).T  # (M, N)

        max_y = int(Y_mat_true.max())
        max_b = int(B_mat.max())
        max_count = max(max_y, max_b)
        pre_gammaln = gammaln(np.arange(max_count + 1) + 1.0)

        const_y_fact = float(np.sum(pre_gammaln[OD_vec]))   # sum log(y!)
        const_u_fact = float(np.sum(pre_gammaln[B_mat]))    # sum log(u!)

        print("  Found Y_mat_true, OD_vec shape:", OD_vec.shape)
    else:
        OD_vec = None
        const_y_fact = None
        const_u_fact = None
        print("  No Y_mat_true, loglikelihood will be NaN.")

    # Temporal kernel & Psi prior
    t = np.arange(N, dtype=float)
    kernel_matrix = combined_kernel(t, t, period, l_p, l_e, w)
    choll = np.linalg.cholesky(kernel_matrix + 1e-6 * np.eye(N))

    # --- Storage: keep-only for y/Phi/Psi; all-iter for loglikelihood ---
    y_store_keep   = np.zeros((n_keep, M, N), dtype=np.int32)
    Phi_store_keep = np.zeros((n_keep, M, D), dtype=np.float64)
    Psi_store_keep = np.zeros((n_keep, N, D), dtype=np.float64)

    loglikelihood_all = np.full(epoch, np.nan, dtype=np.float64)

    keep_ptr = 0
    next_keep = keep_idx[keep_ptr] if n_keep > 0 else -1

    # --- Current state ---
    Y_current = np.zeros((N, S, S), dtype=np.int64)
    y_vec_current = np.zeros((M, N), dtype=np.int32)

    # Initialize Phi, Psi
    Phi = np.zeros((M, D), dtype=np.float64)
    Psi = choll @ rng.normal(0.0, 1.0, size=(N, D))

    # initial P (needed for Y update at it=0)
    G = compute_G(Phi, Psi)
    P = compute_lambda(G, row_slices)  # (M, N)

    # Detect whether ESS accepts rng
    _ess_has_rng = ("rng" in inspect.signature(elliptical_slice_sampling_Psi).parameters)

    print("  Initialization done, start MCMC...")
    t_start = time.time()

    # ============================================================
    # MCMC iterations
    # ============================================================
    for it in range(epoch):

        # --------------------------------------------------------
        # (a) Update Y for each trip n (uses p_vec directly)
        # --------------------------------------------------------
        acc_Y = 0
        for n in range(N):

            U = B_mat[n]  # boardings
            V = A_mat[n]  # alightings

            p_vec = P[:, n].astype(np.float64)

            #if rng.random() < gamma_tilt:
                #p_vec_prop = tilt_pvec_blockwise( p_vec, row_start_y, row_len_y,alpha=alpha_tilt, eps=eps_tilt)
            #else:
                #p_vec_prop = p_vec
            #W_n = compute_W_from_pvec(p_vec_prop, row_start_y, row_len_y, S)

            # Build W directly from p_vec (NO p_mat)
            W_n = compute_W_from_pvec(p_vec, row_start_y, row_len_y, S)

            if it == 0:
                # init only: proposal draw returns (Y, log_q); we only need Y
                Y0, _logq0 = sample_Y_once(V, U, W_n, rng=rng)
                Y_current[n] = Y0
            else:
                # MH step: your updated mh_step_Y returns (Y_new, accepted)
                Y_old = Y_current[n]
                Y_new, accepted = mh_step_Y(
                    current_Y=Y_old,
                    U=U,
                    V=V,
                    p_vec=p_vec,
                    row_start=row_start_y,
                    row_len=row_len_y,
                    S=S,
                    W=W_n,
                    rng=rng,
                )
                Y_current[n] = Y_new
                if accepted:
                    acc_Y += 1

            # update y-vector for Phi update & storage
            y_vec_current[:, n] = get_upper_right(Y_current[n])

        # --------------------------------------------------------
        # (b) Update Phi (pMALA)
        # --------------------------------------------------------
        Y_for_phi = y_vec_current.T  # (N, M)

        # Optional debug check: ensure Y_for_phi row-margins match u (=B_mat)
        if row_margin_debug and (it % row_margin_check_every == 0):
            ok, bad_i, diff = check_row_margins(Y_for_phi, B_mat, row_slices)
            if not ok:
                bad_n = np.where(diff != 0)[0][:10]
                raise ValueError(
                    f"[Iter {it}] Row-sum mismatch at origin block {bad_i}. "
                    f"diff range: [{diff.min()}, {diff.max()}]. "
                    f"First bad buses: {bad_n}"
                )
        Phi, block_acc_cum, block_trials_cum, block_accepted_now, current_logpost = pmala_update_phi(
            Phi=Phi,
            Psi=Psi,
            Y=Y_for_phi,
            u=B_mat,
            sigma0=sigma0,
            row_slices=row_slices,
            h_phi=h_phi_per_block,        
            rng=rng,
            M_mode="k_scaled",           
            M_inflate=1.0,
            m_min=1e-8,
            h_min=h_min,
            h_max=h_max,
        )

        if it < burnin:
            for i in range(n_blocks):
                accept_rate_ema[i] = ema_decay * accept_rate_ema[i] + (1 - ema_decay) * block_accepted_now[i]
                h_phi_per_block[i] *= np.exp(adapt_gamma * (accept_rate_ema[i] - target_accept))
                h_phi_per_block[i] = np.clip(h_phi_per_block[i], h_min, h_max)


        # --------------------------------------------------------
        # (c) Update Psi (ESS)
        # --------------------------------------------------------
        for d in range(D):
            if _ess_has_rng:
                Psi[:, d] = elliptical_slice_sampling_Psi(
                    d, Psi[:, d], choll, N, Phi, Psi, Y_for_phi, B_mat, row_slices, rng=rng
                )
            else:
                Psi[:, d] = elliptical_slice_sampling_Psi(
                    d, Psi[:, d], choll, N, Phi, Psi, Y_for_phi, B_mat, row_slices
                )

        # --------------------------------------------------------
        # (d) Update P (needed for next iteration's Y)
        # --------------------------------------------------------
        G = compute_G(Phi, Psi)
        P = compute_lambda(G, row_slices)

        # --------------------------------------------------------
        # (e) loglikelihood for ALL iterations (burn-in + sampling)
        # --------------------------------------------------------
        if has_truth:
            loglikelihood_all[it] = (
                float(np.sum(OD_vec * np.log(P + 1e-12)))
                - const_y_fact
                + const_u_fact
            )
        else:
            loglikelihood_all[it] = np.nan

        # --------------------------------------------------------
        # (f) Store ONLY thinned samples
        # --------------------------------------------------------
        if it == next_keep:
            y_store_keep[keep_ptr]   = y_vec_current
            Phi_store_keep[keep_ptr] = Phi
            Psi_store_keep[keep_ptr] = Psi

            keep_ptr += 1
            next_keep = keep_idx[keep_ptr] if keep_ptr < n_keep else -1

        # --------------------------------------------------------
        # (g) Progress
        # --------------------------------------------------------
        if (it + 1) % print_every == 0 or it == 0 or it == epoch - 1:
            print(f"\niteration {it+1}/{epoch}")
            if it == 0:
                print("  MH for Y: initialization only (no MH step)")
            else:
                print(f"  MH for Y: accept {acc_Y}/{N} = {acc_Y / N:.3f}")

            total_acc = block_acc_cum.sum()
            total_trials = block_trials_cum.sum()
            if total_trials > 0:
                ar_cum = total_acc / total_trials
                ar_ema_mean = np.mean(accept_rate_ema)
                print(f"  pMALA for Phi: cum. accept {total_acc}/{total_trials} = {ar_cum:.3f}, EMA accept = {ar_ema_mean:.3f}")

    t_end = time.time()
    print(f"\nTotal running time: {t_end - t_start:.1f} seconds")

    samples = {
        "y_store":       y_store_keep,
        "Phi_store":     Phi_store_keep,
        "Psi_store":     Psi_store_keep,
        # loglikelihood aligned with kept draws:
        "loglikelihood": loglikelihood_all[keep_idx],
    }

    return {
        "samples": samples,
        "keep_idx": keep_idx,
        # full trace of loglikelihood (burn-in + sampling):
        "loglikelihood_all": loglikelihood_all,
        "meta": {
            "S": S,
            "D": D,
            "epoch": epoch,
            "burnin": burnin,
            "thin": thin,
            "seed": seed,
        },
    }

def _run_mcmc_single_chain(args):
    """
    Helper for multiprocessing: run a single chain.
    """
    chain_id, seed, B_mat, A_mat, Y_mat_true, mcmc_kwargs = args

    print(f"[worker] chain {chain_id+1}, seed={seed} start...")
    out = run_mcmc(
        B_mat=B_mat,
        A_mat=A_mat,
        Y_mat_true=Y_mat_true,
        seed=seed,
        **mcmc_kwargs,
    )
    out["chain_id"] = chain_id
    print(f"[worker] chain {chain_id+1} done.")
    return out

def run_many_chains(
    B_mat,
    A_mat,
    Y_mat_true=None,
    n_chains=4,
    base_seed=2025,
    parallel=True,
    n_jobs=None,
    **mcmc_kwargs,
):
    """
    Run multiple MCMC chains (optionally in parallel).

    Extra keyword arguments in mcmc_kwargs are passed to run_mcmc.
    """
    B_mat = np.asarray(B_mat, int)
    A_mat = np.asarray(A_mat, int)
    if Y_mat_true is not None:
        Y_mat_true = np.asarray(Y_mat_true, int)

    if n_jobs is None:
        n_jobs = n_chains

    # pack arguments for each chain
    jobs = []
    for c in range(n_chains):
        seed = base_seed + c
        jobs.append((c, seed, B_mat, A_mat, Y_mat_true, mcmc_kwargs))

    # serial mode (nice for debugging)
    if (not parallel) or (n_chains == 1):
        chains_out_list = []
        for job in jobs:
            chains_out_list.append(_run_mcmc_single_chain(job))
        return chains_out_list

    # parallel mode
    try:
        ctx = mp.get_context("fork")   # Mac/Linux
    except ValueError:
        ctx = mp.get_context("spawn")  # Windows / certain environments

    with ctx.Pool(processes=n_jobs) as pool:
        chains_out_list = pool.map(_run_mcmc_single_chain, jobs)

    return chains_out_list
