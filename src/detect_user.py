from numpy import sum, zeros, array
from .optimisation.descente_gradient import descente_gradient
from math import pow

def distance(weight, a= -50, n=4):
    return pow(10, (a - weight)/(10 * n))

def detect_user_position(A, W, n:int, d = distance, M0=None):
    # definition de la fonction d'érreur
    def f(x):
        obj = 0
        for i in range(n):
            obj += (sum(( x - A[i])**2) - d(W[i])**2)**2
        return obj
        
    # definition du gradient
    def gradf(x):
        dim = A[0].shape[0]
        grad = zeros(dim)
        for i in range(n):
            grad += (sum((x - A[i])**2) - d(W[i])**2)(x - A[i])
        return 4 * grad
    
    # calcul de la position
    if M0 == None :
        x0 = array(M0)
    else:
        x0 = sum(A, axis=0)/n
    M = descente_gradient(f, gradf, x0)
    
    return M
    
def select_user_zone(M, Z) :
    # ici il faut que tu me dise comment tu définis une zone
    return None