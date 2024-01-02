from .search_pas import search_pas


# algorithme de descente de gradient
def descente_gradient(f, gradf, x0, e = 1e-4, max_iters = 10000, verbose = False):
    # position initiale
    x = x0
    objs = [f(x0)]
    for it in range(max_iters) :
        if verbose :
            print("{}\r".format(objs[it]))
        
        # la direction de descente
        d = - gradf(x)
        # le pas de descente 
        s = search_pas(f,gradf,x,d)
        # mise à jour de la nouvelle position
        x = x + s*d
        
        # analyse de convergence
        obj = f(x)
        objs.append(obj)
        if abs(obj - objs[it]) < e*objs[it]:
            break
        
    return x, objs
    