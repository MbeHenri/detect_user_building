import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from src.load import load_collected_user
from src.plot import plot_position

# Parsing des arguments CLI
parser = argparse.ArgumentParser(description="Inférence locale de géolocalisation WiFi")
parser.add_argument(
    "--weights-dir",
    "-w",
    type=str,
    default="outputs",
    help="Répertoire des poids/artefacts locaux (défaut: './outputs')",
)
parser.add_argument(
    "--config-file",
    "-c",
    type=str,
    default="building_config.json",
    help="Fichier de configuration de la carte (défaut: 'building_config.json')",
)
args = parser.parse_args()

if __name__ == "__main__":
    # Chargement de la configuration de carte du bâtiment
    config_path = Path(args.config_file)
    with open(config_path, "r") as f:
        config = json.load(f)

    image_batiment = config["image_batiment"]
    image_curseur = config["image_curseur"]
    echelle = config["echelle"]

    # Chemins des poids locaux
    weights_dir = Path(args.weights_dir)
    premodel_path = weights_dir / "premodel_pipeline.joblib"
    model_path = weights_dir / "DecisionTree.joblib"

    # Chargement des données brutes usager
    USER_DATA_PATH = "./datas/without_acces/BSSID_user.csv"
    usager = load_collected_user(USER_DATA_PATH)

    # Chargement et application du preModel
    print(f"[INFO] Chargement du preModel local : {premodel_path}")
    preModel = joblib.load(premodel_path)
    X_transformed = preModel.transform(usager)

    # Chargement et application du modèle de détection
    print(f"[INFO] Chargement du modèle de détection local : {model_path}")
    detect_model = joblib.load(model_path)

    # Inférence (Prédiction des coordonnées de l'usager)
    M = detect_model.predict(X_transformed)[0]

    # Affichage des résultats
    print("[position détectée]")
    print(f" M (x : {M[0]}, y : {M[1]})")

    # Génération et affichage de la carte de localisation
    image = plot_position(
        M,
        image_batiment,
        image_curseur,
        echelle=echelle,
        showed=False,
    )

    plt.imshow(image)
    plt.axis("off")
    plt.show()
