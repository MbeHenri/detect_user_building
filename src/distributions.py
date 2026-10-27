import json
from pathlib import Path
import numpy as np


def compute_univariate_gaussian_kl(
    mu_test: np.ndarray,
    sigma_test: np.ndarray,
    mu_train: np.ndarray,
    sigma_train: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Cas 1 : Variables indépendantes (diagonales)."""
    sigma_test = np.maximum(sigma_test, eps)
    sigma_train = np.maximum(sigma_train, eps)

    kl_per_feature = (
        np.log(sigma_train / sigma_test)
        + (sigma_test**2 + (mu_test - mu_train) ** 2) / (2 * (sigma_train**2))
        - 0.5
    )

    return float(np.mean(kl_per_feature)), kl_per_feature


def compute_multivariate_gaussian_kl(
    mu_test: np.ndarray,
    cov_test: np.ndarray,
    mu_train: np.ndarray,
    cov_train: np.ndarray,
    reg_factor: float = 1e-6,
) -> float:
    """Cas 2 : Variables dépendantes (corrélées)."""
    d = len(mu_train)

    # Régularisation sur la diagonale pour garantir l'inversibilité
    cov_train_reg = cov_train + reg_factor * np.eye(d)
    cov_test_reg = cov_test + reg_factor * np.eye(d)

    cov_train_inv = np.linalg.inv(cov_train_reg)
    _, logdet_train = np.linalg.slogdet(cov_train_reg)
    _, logdet_test = np.linalg.slogdet(cov_test_reg)

    # 1. Trace
    trace_term = np.trace(cov_train_inv @ cov_test_reg)

    # 2. Terme quadratique
    diff_mu = mu_train - mu_test
    quad_term = (diff_mu.T @ cov_train_inv @ diff_mu).item()

    # 3. Log-déterminant
    logdet_term = logdet_train - logdet_test

    kl_div = 0.5 * (trace_term + quad_term - d + logdet_term)

    return max(0.0, float(kl_div))


def is_diagonal_covariance(cov_matrix: np.ndarray, tol: float = 1e-5) -> bool:
    """Vérifie si la matrice est purement diagonale."""
    off_diag = cov_matrix - np.diag(np.diag(cov_matrix))
    return bool(np.all(np.abs(off_diag) < tol))


def evaluate_and_save_gaussian_kl(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    output_filepath: Path,
    force_mode: str | None = None,
) -> float:
    """Sélectionne automatiquement ou manuellement le mode de calcul et sauvegarde le JSON."""
    mu_train = np.mean(X_train, axis=0)
    cov_train = np.cov(X_train, rowvar=False)

    mu_test = np.mean(X_test, axis=0)
    cov_test = np.cov(X_test, rowvar=False)

    if force_mode == "independent":
        is_independent = True
    elif force_mode == "dependent":
        is_independent = False
    else:
        is_independent = is_diagonal_covariance(cov_train) and is_diagonal_covariance(
            cov_test
        )

    if is_independent:
        sigma_train = np.std(X_train, axis=0)
        sigma_test = np.std(X_test, axis=0)

        mean_kl, kl_per_feature = compute_univariate_gaussian_kl(
            mu_test=mu_test,
            sigma_test=sigma_test,
            mu_train=mu_train,
            sigma_train=sigma_train,
        )

        gaussian_params = {
            "mode": "independent_univariate",
            "mean_kl_divergence_Ptest_Ptrain": mean_kl,
            "train_distribution": {
                "mu": mu_train.tolist(),
                "sigma": sigma_train.tolist(),
            },
            "test_distribution": {
                "mu": mu_test.tolist(),
                "sigma": sigma_test.tolist(),
            },
            "kl_per_feature": {
                str(name): float(kl) for name, kl in zip(feature_names, kl_per_feature)
            },
        }
    else:
        mean_kl = compute_multivariate_gaussian_kl(
            mu_test=mu_test,
            cov_test=cov_test,
            mu_train=mu_train,
            cov_train=cov_train,
        )

        gaussian_params = {
            "mode": "dependent_multivariate",
            "mean_kl_divergence_Ptest_Ptrain": mean_kl,
            "feature_names": [str(f) for f in feature_names],
            "train_distribution": {
                "mu": mu_train.tolist(),
                "cov": cov_train.tolist(),
            },
            "test_distribution": {
                "mu": mu_test.tolist(),
                "cov": cov_test.tolist(),
            },
        }

    with open(output_filepath, "w") as f:
        json.dump(gaussian_params, f, indent=4)

    return mean_kl
