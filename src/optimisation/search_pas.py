from numpy import dot

# fonction qui permet de trouver un pas suffisament grand et petit pour atteindre la convergence
def search_pas(f, gradf, x, d, s0 = 1e-4, smax = 1 , smin = 0):
    """ s = s0
    correct, cond = cond_wolfe(s, f, gradf, x, d)
    while not correct :
        if cond == 1 :
            # le pas est trop grand
            smax = s
        else:
            # le pas est trop petit
            smin = s
        s = 0.5 * (smax + smin) """
    return s0

# fonction permettant de verifier les conditions de wolfe
# qui permettent de vérifier qu'un pas donné soit suffisament grand et petit
# pour s'aasurer d'une bonne convergence
def cond_wolfe(s, f, gradf, x, d, e1= 1e-04, e2=0.99):
    # 0 < e1 < e2 < 1
    x_sd = x + s*d
    prod = dot(gradf(x), d)
    
    dif = f(x_sd) - f(x) - e1*s*prod 
    if dif > 0 :
        return False, 1
        
    dif = dot(gradf(x_sd), d) - e2*prod
    if dif < 0 :
        return False, 2
    
    return True, 0

