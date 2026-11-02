import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from src.load import load_signal_pc
from src.plot import plot_position

# Configuration des arguments CLI
parser = argparse.ArgumentParser(
    description="Script de géolocalisation WiFi",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""[EXEMPLES]
  python main.py -c     # Localisation par similarité
  python main.py -ca    # Localisation par calcul des points d'accès
  python main.py -ia    # Localisation par Machine Learning (modèle legacy local)
""",
)

# Groupe d'options mutuellement exclusives
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "-c",
    "--similarity",
    action="store_true",
    help="Méthode par similarité (signaux collectés avec leurs positions)",
)
mode_group.add_argument(
    "-ca",
    "--access-points",
    action="store_true",
    help="Méthode par calcul de la position des points d'accès",
)
mode_group.add_argument(
    "-ia",
    "--ml-inference",
    action="store_true",
    help="Méthode par apprentissage automatique (Inférence avec modèle local)",
)

parser.add_argument(
    "--config-file",
    "-cfg",
    type=str,
    default="building_config.json",
    help="Fichier de configuration de la carte (défaut: 'building_config.json')",
)
parser.add_argument(
    "--weights-dir",
    "-w",
    type=str,
    default="outputs",
    help="Répertoire des poids locaux pour l'option -ia (défaut: './outputs')",
)


def main():
    args = parser.parse_args()

    # 1. Chargement de la configuration de la carte du bâtiment
    config_path = Path(args.config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    image_batiment = config["image_batiment"]
    image_curseur = config["image_curseur"]
    echelle = config["echelle"]

    # 2. Chargement des signaux bruts de l'usager connecté au PC
    usager = load_signal_pc()

    # --- OPTION 1 : Similarité (-c) ---
    if args.similarity:
        from src.detect_user import detect_collected_sim
        from src.load import load_collected_positions_2d

        P, D = load_collected_positions_2d("./datas/without_acces/BSSIDs_collected.csv")
        M = detect_collected_sim(usager, P, D, k=1)

        if M is not None:
            print("[Position détectée - Similarité]")
            print(f" M (x : {M[0]}, y : {M[1]})")

            plot_position(
                M,
                image_batiment,
                image_curseur,
                echelle=echelle,
                path_computed="./position_compute.png",
            )
        else:
            print("[Aucune position détectée]")

    # --- OPTION 2 : Calcul des points d'accès (-ca) ---
    elif args.access_points:
        from src.detect_access_points import detect_collected_access
        from src.detect_user import detect_collected_compute_access
        from src.load import (
            load_acces_points_save_2d,
            load_to_compute_acces_points_2d,
        )
        from src.plot import plot_access
        from src.save import save_acces_points_2d

        access_points_path = Path("./datas/without_acces/access_points_compute.csv")

        try:
            access_points = load_acces_points_save_2d(str(access_points_path))
        except Exception:
            C = load_to_compute_acces_points_2d(
                "./datas/without_acces/BSSIDs_collected.csv"
            )
            access_points = detect_collected_access(C)
            save_acces_points_2d(access_points, str(access_points_path))

        M = detect_collected_compute_access(usager, access_points)

        if M is not None:
            print("[Position détectée - Points d'accès]")
            print(f" M (x : {M[0]}, y : {M[1]})")

            img_acc = plot_access(
                access_points,
                None,
                "./images/signal.png",
                image=image_batiment,
                echelle=echelle,
                path_computed="./access_points_compute.png",
                showed=False,
            )

            plot_position(
                M,
                img_acc,
                image_curseur,
                echelle=echelle,
                path_computed="./position_compute.png",
            )
        else:
            print("[Aucune position détectée]")

    # --- OPTION 3 : Apprentissage Automatique Local (-ia) ---
    elif args.ml_inference:
        weights_dir = Path(args.weights_dir)
        premodel_path = weights_dir / "premodel_pipeline.joblib"
        model_path = weights_dir / "DecisionTree.joblib"

        if not premodel_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Artefacts d'inférence introuvables dans '{weights_dir}'. "
                "Exécute d'abord 'python legacy_train.py'."
            )

        # 1. Prétraitement
        print(f"[INFO] Chargement du preModel local : {premodel_path}")
        preModel = joblib.load(premodel_path)
        X_transformed = preModel.transform(usager)

        # 2. Modèle de détection
        print(f"[INFO] Chargement du modèle de détection local : {model_path}")
        detect_model = joblib.load(model_path)

        # Inférence
        M = detect_model.predict(X_transformed)[0]

        print("[Position détectée - ML Local]")
        print(f" M (x : {M[0]}, y : {M[1]})")

        plot_position(
            M,
            image_batiment,
            image_curseur,
            echelle=echelle,
            path_computed="./position_compute.png",
        )


if __name__ == "__main__":
    main()
