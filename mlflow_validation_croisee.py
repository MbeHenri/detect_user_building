import argparse
from pathlib import Path
import joblib
import mlflow
from numpy import mean
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.load import load_collected_positions_2d
from src.perfs import mae, mse, rmse

# Gestion de l'option pour le dossier de sortie via la ligne de commande
parser = argparse.ArgumentParser(description="Validation Croisee et suivi MLflow")
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="outputs",
    help="Répertoire de sortie pour les artefacts enregistrés localement (défaut: 'outputs')",
)
args = parser.parse_args()

# Création du dossier de sortie s'il n'existe pas
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    # Configuration de MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Detect_building_xfold")

    # Récupération des positions et des signaux collectés
    P, D = load_collected_positions_2d("datas/without_acces/BSSIDs_collected.csv")

    preModel = Pipeline(
        [("DicVect", DictVectorizer(sparse=False)), ("Scaler", StandardScaler())]
    )
    datas = preModel.fit_transform(D)

    # Définition des modèles de régression
    def DecisionTree():
        return DecisionTreeRegressor()

    def KNN():
        return KNeighborsRegressor()

    regressors = {"KNN": KNN, "DecisionTree": DecisionTree}

    # Définition des fonctions d'évaluation
    metrics = {"rmse": rmse, "mse": mse, "mae": mae, "r2_score": r2_score}

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    perfs = {
        name_model: {name_metric: [] for name_metric in metrics.keys()}
        for name_model in regressors.keys()
    }

    # 1. Enregistrer le Pipeline de prétraitement global comme un artefact partagé
    with mlflow.start_run(run_name="Pre-processing_Pipeline"):
        weights_pretrained_filename = output_dir / "premodel_pipeline.joblib"
        joblib.dump(preModel, weights_pretrained_filename)
        mlflow.log_artifact(str(weights_pretrained_filename))

    # Entraînement et évaluation des modèles avec validation croisée
    for name_model, model_func in regressors.items():
        with mlflow.start_run(run_name=f"global_{name_model}") as parent_run:
            mlflow.log_param("model_type", f"{name_model}")

            table_data = {"Fold": [], "RMSE": [], "MSE": [], "MAE": [], "R2": []}

            # Boucle de validation croisée
            for fold, (train_index, test_index) in enumerate(kf.split(datas)):
                X_train, X_test = datas[train_index], datas[test_index]
                y_train, y_test = P[train_index], P[test_index]

                model_name = f"{name_model}_fold_{fold}"
                with mlflow.start_run(run_name=model_name, nested=True):
                    regressor = model_func()
                    model = regressor.fit(X_train, y_train)

                    # log du modèle avec MLflow
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        name=model_name,
                        input_example=X_train,
                        serialization_format="cloudpickle",
                    )

                    # Inspecter le modèle loggé et ses propriétés
                    logged_model = mlflow.get_logged_model(model_info.model_id)

                    y_pred = model.predict(X_test)

                    # Calcul des métriques pour le fold courant
                    fold_metrics = {}
                    for name_metric in metrics.keys():
                        metric_value = metrics[name_metric](y_test, y_pred)
                        perfs[name_model][name_metric].append(metric_value)
                        mlflow.log_metric(
                            name_metric, metric_value, model_id=logged_model.model_id
                        )
                        fold_metrics[name_metric] = metric_value

                    table_data["Fold"].append(fold)
                    table_data["RMSE"].append(fold_metrics["rmse"])
                    table_data["MSE"].append(fold_metrics["mse"])
                    table_data["MAE"].append(fold_metrics["mae"])
                    table_data["R2"].append(fold_metrics["r2_score"])

                    # Sauvegarde du modèle dans le répertoire personnalisé
                    weight_filename = output_dir / f"{name_model}_fold_{fold}.joblib"
                    joblib.dump(model, weight_filename)
                    mlflow.log_artifact(
                        str(weight_filename), artifact_path="model_weights"
                    )

            # Calcul et enregistrement des performances moyennes globales
            for name_metric in metrics.keys():
                mean_metric_value = mean(perfs[name_model][name_metric])
                mlflow.log_metric(f"mean_{name_metric}", mean_metric_value)

            # Log du tableau détaillé par fold
            mlflow.log_table(data=table_data, artifact_file="perfs_by_fold.json")

            # Enregistrer également le tableau récapitulatif moyen
            summary_data = {
                "Metric": ["RMSE", "MSE", "MAE", "R2"],
                "Mean_Value": [
                    mean(perfs[name_model]["rmse"]),
                    mean(perfs[name_model]["mse"]),
                    mean(perfs[name_model]["mae"]),
                    mean(perfs[name_model]["r2_score"]),
                ],
            }
            mlflow.log_table(data=summary_data, artifact_file="perfs_summary.json")
