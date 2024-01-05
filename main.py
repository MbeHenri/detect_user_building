import sys
from src.plot import plot_position, openImage


def main(args):
    if len(args) == 1:
        option = args[0]

        ECHELLE = 100
        image_batiment = openImage("./images/batiment.png")

        if option == "-a":
            from src.detect_user import detect_with_acces
            from src.load import load_acces_points_2d

            A, W = load_acces_points_2d("./datas/with_acces/BSSID.csv")
            M = detect_with_acces(A, W)

            print("[position détectée]  ")
            print(f" M (x : {M[0]}, y : {M[1]})")

            plot_position(
                M,
                None,
                "./images/pointeur_violet.png",
                image=image_batiment,
                path_computed="./position_compute.png",
                echelle=ECHELLE,
            )

        elif option == "-c":
            from src.detect_user import detect_collected_sim
            from src.load import load_collected_user, load_collected_positions_2d

            E = load_collected_user("./datas/without_acces/BSSID_user.csv")
            P, D = load_collected_positions_2d(
                "./datas/without_acces/BSSIDs_collected.csv"
            )
            M = detect_collected_sim(E, P, D, k=1)

            if M is not None:
                print("[Position détectée]  ")
                print(f" M (x : {M[0]}, y : {M[1]})")

                plot_position(
                    M,
                    None,
                    "./images/pointeur_violet.png",
                    image=image_batiment,
                    path_computed="./position_compute.png",
                    echelle=ECHELLE,
                )
            else:
                print("[Aucune position détectée]  ")

        elif option == "-ca":
            from src.load import (
                load_to_compute_acces_points_2d,
                load_collected_user,
                load_acces_points_save_2d,
            )
            from src.detect_access_points import detect_collected_access
            from src.detect_user import detect_collected_compute_access
            from src.save import save_acces_points_2d
            from src.plot import plot_access

            E = load_collected_user("./datas/without_acces/BSSID_user.csv")

            try:
                access_points = load_acces_points_save_2d(
                    "./datas/without_acces/access_points_compute.csv"
                )
            except Exception:
                C = load_to_compute_acces_points_2d(
                    "./datas/without_acces/BSSIDs_collected.csv"
                )
                access_points = detect_collected_access(C)
                save_acces_points_2d(
                    access_points, "./datas/without_acces/access_points_compute.csv"
                )

            M, _ = detect_collected_compute_access(E, access_points)

            if M is not None:
                print("[Position détectée]  ")
                print(f" M (x : {M[0]}, y : {M[1]})")

                image_batiment = plot_access(
                    access_points,
                    None,
                    "./images/signal.png",
                    image=image_batiment,
                    echelle=ECHELLE,
                    path_computed="./access_points_compute.png",
                    showed=False,
                )

                plot_position(
                    M,
                    None,
                    "./images/pointeur_violet.png",
                    image=image_batiment,
                    path_computed="./position_compute.png",
                    echelle=ECHELLE,
                )
            else:
                print("[Aucune position détectée]  ")

    else:
        print("[USAGE]")
        print("python main.py <-a|-c|-ca>\n")
        print("-a : option si on a les positions des points d'accès")
        print(
            "-c : option si on a un ensemble de signaux collectés avec leurs positions (méthode par similarité)"
        )
        print(
            "-ca : option si on a un ensemble de signaux collectés avec leurs positions (méthode par calcul des points d'accès ) \n"
        )
        print("[EXEMPLE]")
        print("python main.py -c\n")


if __name__ == "__main__":
    main(sys.argv[1:])
