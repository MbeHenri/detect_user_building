import sys

from src.detect_user import detect_with_acces, detect_collected_sim
from src.load import load_acces_points_2d, load_collected_user, load_collected_positions_2d

def main(args):
    
    if len(args) == 1:
        option = args[0]
        if option == "-a" :
            A, W = load_acces_points_2d("./datas/with_acces/BSSID.csv")
            M = detect_with_acces(A, W)
            
            print("[position détectée]  ")
            print(f" M (x : {M[0]}, y : {M[1]})")
            
        elif option == "-c" :
            E = load_collected_user("./datas/without_acces/BSSID_user.csv")
            P, D = load_collected_positions_2d("./datas/without_acces/BSSIDs_collected.csv")
            M = detect_collected_sim(E, P, D, epsilon=0.25)
            
            if M is not None:
                print("[Position détectée]  ")
                print(f" M (x : {M[0]}, y : {M[1]})")
            else:
                print("[Aucune position détectée]  ")
                
    else:
        print('[USAGE]')
        print('python main.py <-a|-c>\n')
        print('-a : option si on a les positions des points d\'accès \n')
        print('-c : option si on a un ensemble de signaux collectés avec leurs positions \n')
        print('[EXEMPLE]')
        print('python main.py a\n')


if __name__ == "__main__":
    main(sys.argv[1:])