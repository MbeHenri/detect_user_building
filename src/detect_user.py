from numpy import sum, zeros, array
from numpy.random import rand
from .optimisation.descente_gradient import descente_gradient
from math import pow

def distance(weight, a= -50, n=4):
    return pow(10, (a - weight)/(10 * n))

def detect_user_position(A, W, d = distance, M0=None):
    n = A.shape[0]
    # definition de la fonction d'érreur
    def f(x):
        obj = 0
        for i in range(n):
            obj += (sum(( x - A[i])**2) - d(W[i])**2)**2
        return obj
        
    # definition du gradient
    def gradf(x):
        g = zeros(A.shape[1], dtype=float)
        for i in range(n):
            g += (sum((x - A[i])**2) - d(W[i])**2)*(x - A[i])
        return 4 * g
    
    # calcul de la position
    if M0 != None :
        x0 = array(M0)
    else:
        x0 = rand(A.shape[1])
        
    M, objs = descente_gradient(f, gradf, x0)
    
    return M, objs
    
def select_user_zone(M, Z) :
    # ici il faut que tu me dise comment tu définis une zone
    return None