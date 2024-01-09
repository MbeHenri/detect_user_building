from numpy import sum, zeros, array
from numpy.random import rand
from .optimisation.descente_gradient import descente_gradient
from math import pow


def distance(weight: float, a=-50, n=4):
    """Evaluer une distance sur la base de la force d'un signal perçu

    Args:
        weight (float): force du signal
        a (int, optional): seuil. Defaults to -50.
        n (int, optional): paramètre d'étallonage. Defaults to 4.

    Returns:
        float: distance du signal perçu
    """
    return pow(10, (a - weight) / (10 * n))


def detect_position(A, W, d=distance, M0=[10, 9]):
    """Détection de la position de l'usager ou d'un point d'accès en considérant les forces de signaux détectés

    Args:
        A (ndarray): l'ensemble des positions des points d'accès ou des usagers
        W (ndarray): l'ensemble des forces de signaux détectés
        d (function, optional): fonction d'évaluation de distance. Defaults to distance.
        M0 (ndarray, optional): fosition initiale à donner à l'usager ou du point d'accès. Defaults to None.

    Returns:
        ndarray: position de l'usager ou du point d'accès
    """
    n = A.shape[0]

    # definition de la fonction d'érreur
    def f(x):
        obj = 0
        for i in range(n):
            obj += (sum((x - A[i]) ** 2) - d(W[i]) ** 2) ** 2
        return obj

    # definition du gradient
    def gradf(x):
        g = zeros(A.shape[1], dtype=float)
        for i in range(n):
            g += (sum((x - A[i]) ** 2) - d(W[i]) ** 2) * (x - A[i])
        return 4 * g

    # calcul de la position
    if M0 is not None:
        x0 = array(M0, dtype=float)
    else:
        x0 = rand(A.shape[1])

    M, objs = descente_gradient(f, gradf, x0)

    return M, objs
