from numpy import sum, zeros, array, sqrt
from .detect_position import detect_position, distance


def cosine_similarity(E1, E2) -> float:
    """Similarité (Cosinus de similarité) entre deux ensembles de signaux qui sont vues comme des dictionaires où :
        - la clé est l'adresse MAC
        - la valeur est la valeur de la force du signal

    Args:
        E1 (dict): premier ensemble de signaux
        E2 (dict): deuxième ensemble de signaux

    Returns:
        float: similarité entre les deux ensembles
    """
    intersection = set(E1.keys()) & set(E2.keys())
    num = sum(E1[key] * E2[key] for key in intersection)
    norm1 = sum(E1[key] ** 2 for key in E1.keys())
    norm2 = sum(E2[key] ** 2 for key in E2.keys())
    den = sqrt(norm1) * sqrt(norm2)

    return num / den if den != 0 else 0


def detect_collected_sim(E, P, D, epsilon=0.25, sim=cosine_similarity):
    """ "Détection de la position de l'usager en considérant un ensemble de positions issues des signaux collectés

    Args:
        E (dict): ensemble de signaux de l'utilisateur (dictionaire avec pour clé adresse MAC du signal, pour valeur la force du signal)
        P (ndarray): ensemble des positions où les signaux ont été collectés
        D (list): ensemble des signaux par position
        epsilon (float, optional): seuil à partir duquel on considère qu'un position ressemble à une autre, par rapport aux signaux y détectés. Defaults to 0.25.
        sim (_type_, optional): fonction évaluant la similarité entre deux ensembles de signaux. Defaults to cosine_similarity.

    Returns:
        ndarray: position de l'usager
    """
    P = array(P)
    n = P.shape[0]
    dim = P.shape[1]

    M = zeros(dim)
    sm = 0
    for i in range(n):
        s = sim(E, D[i])
        if s >= epsilon:
            M += s * P[i]
            sm += s
    if sm != 0:
        return M / sm

    return None


def detect_collected_compute_access(E, access_positions, d=distance, M0=None):
    # calculer la position de l'usager
    access_commun = set(E.keys()) & set(access_positions.keys())

    A = []
    W = []
    for k in access_commun:
        A.append(access_positions[k])
        W.append(E[k])

    if len(access_commun) != 0:
        A = array(A)
        W = array(W)
        return detect_position(A, W, d=d, M0=M0)
    return None


def detect_with_acces(A, W, d=distance, M0=None):
    """Détection de la position de l'usager en considérant les forces de signaux détectés

    Args:
        A (ndarray): l'ensemble des positions des points d'accès
        W (ndarray): l'ensemble des forces de signaux détectés
        d (function, optional): fonction d'évaluation de distance. Defaults to distance.
        M0 (ndarray, optional): fosition initiale à donner à l'usager ou du point d'accès. Defaults to None.

    Returns:
        ndarray: position de l'usager
    """
    M, _ = detect_position(A, W, d=d, M0=M0)
    return M
