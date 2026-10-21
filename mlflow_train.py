import argparse
from pathlib import Path
import joblib
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.load import load_collected_positions_2d
from src.perfs import mae, mse, rmse

# Gestion des arguments en ligne de commande
parser = argparse.ArgumentParser(description="Entraînement Train/Test et suivi MLflow")
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="outputs",
    help="Répertoire de sortie pour les artefacts sauvegardés localement (défaut: 'outputs')",
)
args = parser.parse_args()

# Création du dossier de sortie s'il n'existe pas
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    # Configuration de MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Detect_building_traintest")

    # Récupération des positions (P) et des signaux collectés (D)
    P, D = load_collected_positions_2d("datas/without_acces/BSSIDs_collected.csv")

    # Découpage simple en Train / Test (80% train, 20% test)
    D_train, D_test, y_train, y_test = train_test_split(
        D, P, test_size=0.2, random_state=42
    )

    # Prétraitement : fit sur le Train, transform sur Train et Test
    preModel = Pipeline(
        [("DicVect", DictVectorizer(sparse=False)), ("Scaler", StandardScaler())]
    )
    X_train = preModel.fit_transform(D_train)
    X_test = preModel.transform(D_test)

    # Enregistrer le Pipeline de prétraitement comme un artefact MLflow
    with mlflow.start_run(run_name="Pre-processing_Pipeline"):
        weights_pretrained_filename = output_dir / "premodel_pipeline.joblib"
        joblib.dump(preModel, weights_pretrained_filename)
        mlflow.log_artifact(str(weights_pretrained_filename))

    # Définition des modèles
    regressors = {
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
    }

    # Définition des métriques
    metrics = {"rmse": rmse, "mse": mse, "mae": mae, "r2_score": r2_score}

    # Entraînement et évaluation pour chaque modèle
    for name_model, model in regressors.items():
        with mlflow.start_run(run_name=f"run_{name_model}"):
            mlflow.log_param("model_type", name_model)

            # Entraînement du modèle
            model.fit(X_train, y_train)

            # Log du modèle dans MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name=name_model,
                input_example=X_train[:5],
                serialization_format="cloudpickle",
            )

            # Prédictions et calcul des métriques
            y_pred = model.predict(X_test)

            eval_results = {}
            for metric_name, metric_func in metrics.items():
                value = metric_func(y_test, y_pred)
                eval_results[metric_name] = value
                mlflow.log_metric(metric_name, value)

            # Sauvegarde locale du modèle dans le répertoire spécifié
            weight_filename = output_dir / f"{name_model}.joblib"
            joblib.dump(model, weight_filename)
            mlflow.log_artifact(str(weight_filename), artifact_path="model_weights")

            # Sauvegarde d'un tableau synthétique des performances
            summary_data = {
                "Metric": list(eval_results.keys()),
                "Value": list(eval_results.values()),
            }
            mlflow.log_table(data=summary_data, artifact_file="perfs_summary.json")
