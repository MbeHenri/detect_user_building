import argparse
import json
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
from src.distributions import evaluate_and_save_gaussian_kl
from src.load import load_collected_positions_2d
from src.perfs import mae, mse, rmse

# Gestion des arguments
parser = argparse.ArgumentParser(description="Entraînement Train/Test et suivi MLflow")
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="outputs",
    help="Répertoire de sortie pour les artefacts (défaut: './outputs')",
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

    # Calcul de la divergence D_KL entre les données de test et d'entraînement
    # et sauvegarde locale du fichier de distribution
    feature_names = preModel.named_steps["DicVect"].get_feature_names_out()
    dist_filepath = output_dir / "gaussian_distributions_kl.json"

    mean_kl = evaluate_and_save_gaussian_kl(
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        output_filepath=dist_filepath,
    )

    # Enregistrement du pipeline et des distributions dans MLflow
    with mlflow.start_run(run_name="Pre-processing_&_Distributions"):
        weights_pretrained_filename = output_dir / "premodel_pipeline.joblib"
        joblib.dump(preModel, weights_pretrained_filename)
        mlflow.log_artifact(str(weights_pretrained_filename))

        mlflow.log_artifact(str(dist_filepath), artifact_path="distributions")
        mlflow.log_metric("dataset_mean_kl_divergence", mean_kl)

    # Définition des modèles et métriques
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

            mlflow.sklearn.log_model(
                sk_model=model,
                name=name_model,
                input_example=X_train[:5],
                serialization_format="cloudpickle",
            )

            # Prédictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Evaluation des métriques
            summary_table = {}
            for metric_name, metric_func in metrics.items():
                perf_train = float(metric_func(y_train, y_pred_train))
                perf_test = float(metric_func(y_test, y_pred_test))

                # calcul de la sensibilité par rapport à la divergence KL
                delta_perf = perf_train - perf_test
                sensitivity = delta_perf / mean_kl if mean_kl != 0 else 0.0

                mlflow.log_metric(f"{metric_name}_train", perf_train)
                mlflow.log_metric(f"{metric_name}_test", perf_test)
                mlflow.log_metric(f"{metric_name}_delta", delta_perf)
                mlflow.log_metric(f"{metric_name}_sensitivity", sensitivity)

                summary_table[metric_name] = {
                    "train": perf_train,
                    "test": perf_test,
                    "delta": delta_perf,
                    "sensitivity": sensitivity,
                }

            # Sauvegarde locale du fichier JSON des performances
            summary_filepath = output_dir / f"{name_model}_perfs_summary.json"
            with open(summary_filepath, "w") as f:
                json.dump(summary_table, f, indent=4)

            # Formatage des données pour MLflow et sauvegarde dans MLflow
            mlflow_table_format = {
                "Metric": list(summary_table.keys()),
                "Perf_Train": [v["train"] for v in summary_table.values()],
                "Perf_Test": [v["test"] for v in summary_table.values()],
                "Delta_Perf": [v["delta"] for v in summary_table.values()],
                "Sensitivity": [v["sensitivity"] for v in summary_table.values()],
            }
            mlflow.log_table(
                data=mlflow_table_format, artifact_file="perfs_summary.json"
            )
            mlflow.log_artifact(str(summary_filepath), artifact_path="metrics_summary")

            # sauvegarde du modèle dans MLflow
            weight_filename = output_dir / f"{name_model}.joblib"
            joblib.dump(model, weight_filename)
            mlflow.log_artifact(str(weight_filename), artifact_path="model_weights")
