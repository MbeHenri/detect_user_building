
from .detect_position import detect_position, distance

def detect_collected_access(C, d=distance, M0=None):
    
    # calculer les positions des points d'accès
    access_positions = {}
    for k in C.keys():
        A, W = C[k]["A"], C[k]["W"]
        access_positions[k], objs = detect_position(A, W, d=d, M0=M0)
        #print(" MAC : {}, Position : {} : niters : {} \n".format(k , access_positions[k], len(objs)))
        
    return access_positions