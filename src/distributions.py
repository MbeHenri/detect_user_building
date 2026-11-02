import json
from pathlib import Path
import numpy as np
import scipy.stats as stats

# =====================================================================
# 1. CONSTRUCTION DES HISTOGRAMMES & PROBABILITÉS
# =====================================================================


def build_histogram_with_overflow(
    values: np.ndarray,
    bins: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Construit l'histogramme normalisé sur la grille bins du Train avec débordement :
    index 0 (< min Train) et index -1 (> max Train).
    """
    c_inner, _ = np.histogram(values, bins=bins)
    c_below = np.sum(values < bins[0])
    c_above = np.sum(values > bins[-1])

    counts = np.concatenate([[c_below], c_inner, [c_above]]).astype(float)
    p = counts / max(counts.sum(), 1.0)
    p = np.maximum(p, eps)
    p /= p.sum()

    return p


def compute_bin_indices_with_overflow(
    values: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """Retourne l'index de bac avec gestion du débordement."""
    n_bins = len(bins) - 1
    bin_indices = np.digitize(values, bins)
    bin_indices[values == bins[-1]] = n_bins

    return bin_indices


def compute_probability_gaussian(
    X: np.ndarray,
    mu_train: np.ndarray,
    sigma_train: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Calcule la probabilité discrétisée P(x) par intégration CDF Gaussienne 1D."""
    sigma_train = np.maximum(sigma_train, eps)
    n_samples, d = X.shape
    p_matrix = np.zeros((n_samples, d))

    for f_idx in range(d):
        mu = mu_train[f_idx]
        sigma = sigma_train[f_idx]
        x_vals = X[:, f_idx]

        dx = sigma * 0.05
        cdf_upper = stats.norm.cdf(x_vals + dx, loc=mu, scale=sigma)
        cdf_lower = stats.norm.cdf(x_vals - dx, loc=mu, scale=sigma)

        p_feat = cdf_upper - cdf_lower
        p_matrix[:, f_idx] = np.maximum(p_feat, eps)

    return p_matrix


def compute_probability_histogram(
    X: np.ndarray,
    hist_train: list[np.ndarray],
    bins_train: list[np.ndarray],
    eps: float = 1e-10,
) -> np.ndarray:
    """Calcule la probabilité empirique P(x) en la projetant sur les histogrammes du Train."""
    n_samples, d = X.shape
    p_matrix = np.zeros((n_samples, d))

    for f_idx in range(d):
        x_vals = X[:, f_idx]
        p_train = hist_train[f_idx]
        bins = bins_train[f_idx]

        bin_indices = compute_bin_indices_with_overflow(x_vals, bins)
        p_obs = p_train[bin_indices]
        p_matrix[:, f_idx] = np.maximum(p_obs, eps)

    return p_matrix


# =====================================================================
# 2. CALCUL DES DIVERGENCES KL
# =====================================================================


def compute_kl_gaussian(
    mu_test: np.ndarray,
    sigma_test: np.ndarray,
    mu_train: np.ndarray,
    sigma_train: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Calcul analytique de D_KL(P_test || P_train) pour Gaussiennes indépendantes."""
    sigma_test = np.maximum(sigma_test, eps)
    sigma_train = np.maximum(sigma_train, eps)

    kl_per_feature = (
        np.log(sigma_train / sigma_test)
        + (sigma_test**2 + (mu_test - mu_train) ** 2) / (2 * (sigma_train**2))
        - 0.5
    )
    return float(np.mean(kl_per_feature)), kl_per_feature


def compute_kl_histogram(
    hist_test: list[np.ndarray],
    hist_train: list[np.ndarray],
    eps: float = 1e-10,
) -> tuple[float, np.ndarray]:
    """Calcul de D_KL(P_test || P_train) basé sur les histogrammes empiriques."""
    kl_per_feature = []

    for p_te, p_tr in zip(hist_test, hist_train):
        p_te = np.maximum(p_te, eps)
        p_tr = np.maximum(p_tr, eps)

        kl_f = np.sum(p_te * np.log(p_te / p_tr))
        kl_per_feature.append(max(0.0, float(kl_f)))

    kl_array = np.array(kl_per_feature)
    return float(np.mean(kl_array)), kl_array


# =====================================================================
# 3. CALCUL DE LA SURPRISE
# =====================================================================


def compute_surprisal(p_matrix: np.ndarray) -> np.ndarray:
    """Calcule la surprise s(x) = -log2(P(x)) par feature puis somme sur les features."""
    feat_surprisals = -np.log2(p_matrix)
    return np.sum(feat_surprisals, axis=1)


def compute_surprisal_stats(obs_surprisals: np.ndarray) -> dict[str, float]:
    """Calcule les statistiques globales de la surprise."""
    return {
        "surprisal_mean": float(np.mean(obs_surprisals)),
        "surprisal_median": float(np.median(obs_surprisals)),
        "surprisal_std": float(np.std(obs_surprisals)),
    }


# =====================================================================
# 4. FONCTION PRINCIPALE POUR MLFLOW
# =====================================================================


def evaluate_and_save_gaussian_kl(
    X_train: np.ndarray,
    X_test: np.ndarray,
    output_filepath: Path,
    n_bins: int = 20,
    eps: float = 1e-10,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Construit les distributions (moy, cov, sigma, min, max, hist), évalue D_KL et la Surprise
    pour les approches Gaussienne et Histogramme (indexées par position), puis sauvegarde le JSON.
    """
    d = X_train.shape[1]

    # --- 1. Statistiques Train ---
    mu_train = np.mean(X_train, axis=0)
    cov_train = np.cov(X_train, rowvar=False)
    sigma_train = np.std(X_train, axis=0, ddof=1)
    min_train = np.min(X_train, axis=0)
    max_train = np.max(X_train, axis=0)

    # --- 2. Statistiques Test ---
    mu_test = np.mean(X_test, axis=0)
    cov_test = np.cov(X_test, rowvar=False)
    sigma_test = np.std(X_test, axis=0, ddof=1)
    min_test = np.min(X_test, axis=0)
    max_test = np.max(X_test, axis=0)

    # --- 3. Histogrammes avec débordement ---
    hists_train = []
    hists_test = []
    bins_train_list = []

    for f_idx in range(d):
        tr_f = X_train[:, f_idx]
        te_f = X_test[:, f_idx]

        mn_tr, mx_tr = min_train[f_idx], max_train[f_idx]
        if mn_tr == mx_tr:
            bins = np.linspace(mn_tr - 1e-5, mx_tr + 1e-5, n_bins + 1)
        else:
            bins = np.linspace(mn_tr, mx_tr, n_bins + 1)

        bins_train_list.append(bins)
        hists_train.append(build_histogram_with_overflow(tr_f, bins, eps=eps))
        hists_test.append(build_histogram_with_overflow(te_f, bins, eps=eps))

    # --- 4. Approche Gaussienne ---
    kl_gaussian_mean, kl_gaussian_per_feat = compute_kl_gaussian(
        mu_test=mu_test,
        sigma_test=sigma_test,
        mu_train=mu_train,
        sigma_train=sigma_train,
    )
    p_gaussian = compute_probability_gaussian(
        X=X_test,
        mu_train=mu_train,
        sigma_train=sigma_train,
    )
    obs_surp_gaussian = compute_surprisal(p_gaussian)
    surprisal_gaussian_stats = compute_surprisal_stats(obs_surp_gaussian)

    # --- 5. Approche Histogramme ---
    kl_hist_mean, kl_hist_per_feat = compute_kl_histogram(
        hist_test=hists_test,
        hist_train=hists_train,
    )
    p_hist = compute_probability_histogram(
        X=X_test,
        hist_train=hists_train,
        bins_train=bins_train_list,
    )
    obs_surp_hist = compute_surprisal(p_hist)
    surprisal_hist_stats = compute_surprisal_stats(obs_surp_hist)

    # --- 6. Sauvegarde JSON (Indexation par entier 0 à d-1) ---
    distributions_summary = {
        "n_bins": n_bins,
        "n_features": d,
        "metrics_summary": {
            "gaussian": {
                "kl_divergence": kl_gaussian_mean,
                "surprisal": surprisal_gaussian_stats,
            },
            "histogram": {
                "kl_divergence": kl_hist_mean,
                "surprisal": surprisal_hist_stats,
            },
        },
        "train_distribution": {
            "mu": mu_train.tolist(),
            "sigma": sigma_train.tolist(),
            "cov": cov_train.tolist(),
            "min": min_train.tolist(),
            "max": max_train.tolist(),
            "bins": [b.tolist() for b in bins_train_list],
            "histograms": [h.tolist() for h in hists_train],
        },
        "test_distribution": {
            "mu": mu_test.tolist(),
            "sigma": sigma_test.tolist(),
            "cov": cov_test.tolist(),
            "min": min_test.tolist(),
            "max": max_test.tolist(),
            "histograms": [h.tolist() for h in hists_test],
        },
        "kl_per_feature": {
            "gaussian": kl_gaussian_per_feat.tolist(),
            "histogram": kl_hist_per_feat.tolist(),
        },
    }
    with open(output_filepath, "w") as f:
        json.dump(distributions_summary, f, indent=4)

    kl_results = {
        "gaussian": kl_gaussian_mean,
        "histogram": kl_hist_mean,
    }
    surprisal_results = {
        "gaussian": surprisal_gaussian_stats,
        "histogram": surprisal_hist_stats,
    }

    return kl_results, surprisal_results
