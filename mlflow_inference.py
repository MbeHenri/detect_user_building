import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.load import load_collected_user
from src.plot import plot_position

# Parsing des arguments CLI (Infrastructure & Config)
parser = argparse.ArgumentParser(description="Inférence MLflow de géolocalisation WiFi")
parser.add_argument(
    "--config-file",
    "-c",
    type=str,
    default="building_config.json",
    help="Fichier de configuration de la carte (défaut: 'building_config.json')",
)
parser.add_argument(
    "--tracking-uri",
    "-t",
    type=str,
    default="http://127.0.0.1:5000",
    help="URI du serveur MLflow (défaut: 'http://127.0.0.1:5000')",
)
args = parser.parse_args()

if __name__ == "__main__":
    MODEL_NAME = "WiFi_Building_Detector_DecisionTree"

    # Chargement de la configuration de carte du bâtiment
    config_path = Path(args.config_file)
    with open(config_path, "r") as f:
        config = json.load(f)

    image_batiment = config["image_batiment"]
    image_curseur = config["image_curseur"]
    echelle = config["echelle"]

    # Connexion au serveur MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Chargement des données brutes usager
    USER_DATA_PATH = "./datas/without_acces/BSSID_user.csv"
    usager = load_collected_user(USER_DATA_PATH)

    # Récupération de la dernière version du modèle enregistré
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not model_versions:
        raise ValueError(f"Aucune version trouvée pour le modèle '{MODEL_NAME}'.")

    latest_model_version = sorted(
        model_versions, key=lambda v: int(v.version), reverse=True
    )[0]
    print(
        f"[INFO] Modèle '{MODEL_NAME}' trouvé (Version: {latest_model_version.version}, Run ID: {latest_model_version.run_id})"
    )

    # Chargement du Pipeline complet unifié depuis MLflow Registry
    model_uri = f"models:/{MODEL_NAME}/{latest_model_version.version}"
    print(f"[INFO] Chargement du Pipeline complet depuis MLflow : {model_uri}")
    full_pipeline = mlflow.sklearn.load_model(model_uri)

    # Extraction explicite des deux étapes (preModel puis regressor)
    preModel = full_pipeline.named_steps["preModel"]
    detect_model = full_pipeline.named_steps["regressor"]

    # Exécution séquentielle : Pré-traitement puis Détection
    print("[INFO] Application du preModel...")
    X_transformed = preModel.transform(usager)

    print("[INFO] Prédiction avec le modèle de détection...")
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
