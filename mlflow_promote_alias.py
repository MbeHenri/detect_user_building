import argparse
import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser(
    description="Met à jour l'alias d'un modèle dans MLflow Registry depuis un fichier JSON"
)
parser.add_argument(
    "--metadata-file",
    "-m",
    type=str,
    required=True,
    help="Chemin vers le fichier JSON de métadonnées (ex: outputs/DecisionTree_run_info.json)",
)
parser.add_argument(
    "--alias",
    "-a",
    type=str,
    default="production",
    help="Nom de l'alias à attribuer (défaut: 'production')",
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
    metadata_path = Path(args.metadata_file)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Le fichier de métadonnées '{metadata_path}' n'existe pas."
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    registered_model_name = metadata["registered_model_name"]
    model_version = metadata["model_version"]

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    print(
        f"[INFO] Attribution de l'alias '{args.alias}' "
        f"au modèle '{registered_model_name}' (Version: {model_version})..."
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=args.alias,
        version=model_version,
    )

    print(f"[SUCCÈS] Alias '{args.alias}' attribué avec succès !")
