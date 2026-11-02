import argparse
from datetime import datetime
import json
from pathlib import Path
import joblib
import mlflow
from mlflow.models.signature import infer_signature
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
parser.add_argument(
    "--n-bins",
    type=int,
    default=20,
    help="Nombre de bins pour la discrétisation par histogramme (défaut: 20)",
)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    # Date de l'entrainement
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    # Calcul des distributions, KL et Surprisals
    dist_filepath = output_dir / "distributions_and_kl.json"

    kl_results, surprisal_results = evaluate_and_save_gaussian_kl(
        X_train=X_train,
        X_test=X_test,
        output_filepath=dist_filepath,
        n_bins=args.n_bins,
    )

    # Run de Prétraitement
    with mlflow.start_run(
        run_name=f"Pre-processing_&_Distributions_{timestamp}"
    ) as prep_run:
        prep_run_id = prep_run.info.run_id

        weights_pretrained_filename = output_dir / "premodel_pipeline.joblib"
        joblib.dump(preModel, weights_pretrained_filename)
        mlflow.log_artifact(str(weights_pretrained_filename), artifact_path="premodel")

        mlflow.log_artifact(str(dist_filepath), artifact_path="distributions")

        # 1. Divergences KL
        mlflow.log_metric("kl_divergence_gaussian", kl_results["gaussian"])
        mlflow.log_metric("kl_divergence_histogram", kl_results["histogram"])

        # 2. Statistiques de Surprise Gaussienne
        mlflow.log_metric(
            "test_surprisal_gaussian_mean",
            surprisal_results["gaussian"]["surprisal_mean"],
        )
        mlflow.log_metric(
            "test_surprisal_gaussian_median",
            surprisal_results["gaussian"]["surprisal_median"],
        )
        mlflow.log_metric(
            "test_surprisal_gaussian_std",
            surprisal_results["gaussian"]["surprisal_std"],
        )

        # 3. Statistiques de Surprise Histogramme
        mlflow.log_metric(
            "test_surprisal_histogram_mean",
            surprisal_results["histogram"]["surprisal_mean"],
        )
        mlflow.log_metric(
            "test_surprisal_histogram_median",
            surprisal_results["histogram"]["surprisal_median"],
        )
        mlflow.log_metric(
            "test_surprisal_histogram_std",
            surprisal_results["histogram"]["surprisal_std"],
        )

        # Sauvegarde locale JSON pour le preModel
        premodel_info = {
            "run_id": prep_run_id,
            "artifact_path": "premodel/premodel_pipeline.joblib",
            "timestamp": timestamp,
        }
        premodel_info_file = output_dir / "premodel_info.json"
        with open(premodel_info_file, "w") as f:
            json.dump(premodel_info, f, indent=4)

        mlflow.log_artifact(str(premodel_info_file), artifact_path="metadata")

    # Définition et évaluation des régresseurs
    regressors = {
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
    }

    # Définition des métriques
    metrics = {"rmse": rmse, "mse": mse, "mae": mae, "r2_score": r2_score}

    # Entraînement et évaluation pour chaque modèle
    for name_model, model in regressors.items():
        run_name = f"run_{name_model}_{timestamp}"
        registered_model_name = f"WiFi_Building_Detector_{name_model}"

        with mlflow.start_run(run_name=run_name) as model_run:
            current_run_id = model_run.info.run_id

            mlflow.set_tags(
                {
                    "model_family": name_model,
                    "framework": "scikit-learn",
                    "timestamp": timestamp,
                    "premodel_run_id": prep_run_id,
                }
            )
            mlflow.log_param("model_type", name_model)

            # Entraînement du modèle seul
            model.fit(X_train, y_train)

            # Création du Pipeline complet (Prétraitement + Modèle)
            full_pipeline = Pipeline([("preModel", preModel), ("regressor", model)])

            # Prédictions d'exemple pour déduire explicitement la signature MLflow
            sample_input = D_test[:2]
            sample_pred = full_pipeline.predict(sample_input)
            signature = infer_signature(sample_input, sample_pred)

            # Enregistrement dans le Model Registry MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                name=f"{name_model}_{timestamp}",
                registered_model_name=registered_model_name,
                signature=signature,
                serialization_format="cloudpickle",
            )

            # Prédictions globales
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Évaluation des métriques
            summary_table = {}
            for metric_name, metric_func in metrics.items():
                perf_train = float(metric_func(y_train, y_pred_train))
                perf_test = float(metric_func(y_test, y_pred_test))

                # Calcul de la sensibilité par rapport à la divergence KL
                delta_perf = perf_train - perf_test

                sensitivity_gaussian = (
                    delta_perf / kl_results["gaussian"]
                    if kl_results["gaussian"] != 0
                    else 0.0
                )
                sensitivity_histogram = (
                    delta_perf / kl_results["histogram"]
                    if kl_results["histogram"] != 0
                    else 0.0
                )

                mlflow.log_metric(f"{metric_name}_train", perf_train)
                mlflow.log_metric(f"{metric_name}_test", perf_test)
                mlflow.log_metric(f"{metric_name}_delta", delta_perf)
                mlflow.log_metric(
                    f"{metric_name}_sensitivity_gaussian", sensitivity_gaussian
                )
                mlflow.log_metric(
                    f"{metric_name}_sensitivity_histogram", sensitivity_histogram
                )

                summary_table[metric_name] = {
                    "train": perf_train,
                    "test": perf_test,
                    "delta": delta_perf,
                    "sensitivity_gaussian": sensitivity_gaussian,
                    "sensitivity_histogram": sensitivity_histogram,
                }

            # Artefacts locaux
            summary_filepath = output_dir / f"{name_model}_perfs_summary.json"
            with open(summary_filepath, "w") as f:
                json.dump(summary_table, f, indent=4)

            # Formatage des données pour MLflow et sauvegarde dans MLflow
            mlflow_table_format = {
                "Metric": list(summary_table.keys()),
                "Perf_Train": [v["train"] for v in summary_table.values()],
                "Perf_Test": [v["test"] for v in summary_table.values()],
                "Delta_Perf": [v["delta"] for v in summary_table.values()],
                "Sens_Gaussian": [
                    v["sensitivity_gaussian"] for v in summary_table.values()
                ],
                "Sens_Histogram": [
                    v["sensitivity_histogram"] for v in summary_table.values()
                ],
            }
            mlflow.log_table(
                data=mlflow_table_format, artifact_file="perfs_summary.json"
            )
            mlflow.log_artifact(str(summary_filepath), artifact_path="metrics_summary")

            # Sauvegarde locale du modèle de détection seul
            model_only_filename = output_dir / f"{name_model}.joblib"
            joblib.dump(model, model_only_filename)
            mlflow.log_artifact(str(model_only_filename), artifact_path="model_weights")

            # Sauvegarde locale de la combinaison Pipeline complet (ex: KNN_full_pipeline.joblib)
            full_pipeline_filename = output_dir / f"{name_model}_full_pipeline.joblib"
            joblib.dump(full_pipeline, full_pipeline_filename)
            mlflow.log_artifact(
                str(full_pipeline_filename), artifact_path="full_pipeline_weights"
            )

            # Création du fichier JSON de métadonnées générique pour le modèle
            run_metadata = {
                "model_name": name_model,
                "registered_model_name": registered_model_name,
                "model_version": model_info.registered_model_version,
                "run_id": current_run_id,
                "premodel_run_id": prep_run_id,
                "timestamp": timestamp,
            }
            run_info_filepath = output_dir / f"{name_model}_run_info.json"
            with open(run_info_filepath, "w") as f:
                json.dump(run_metadata, f, indent=4)

            mlflow.log_artifact(str(run_info_filepath), artifact_path="metadata")
