import argparse
import json
from pathlib import Path
import joblib
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
parser = argparse.ArgumentParser(description="Entraînement Train/Test simple")
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
    # Récupération des positions (P) et des signaux collectés (D)
    P, D = load_collected_positions_2d("datas/without_acces/BSSIDs_collected.csv")

    # Découpage Train / Test (80% train, 20% test)
    D_train, D_test, y_train, y_test = train_test_split(
        D, P, test_size=0.2, random_state=42
    )

    # Prétraitement : fit sur le Train, transform sur Train et Test
    preModel = Pipeline(
        [("DicVect", DictVectorizer(sparse=False)), ("Scaler", StandardScaler())]
    )
    X_train = preModel.fit_transform(D_train)
    X_test = preModel.transform(D_test)

    # Sauvegarde locale du preModel seul
    weights_pretrained_filename = output_dir / "premodel_pipeline.joblib"
    joblib.dump(preModel, weights_pretrained_filename)
    print(f"[INFO] preModel sauvegardé : {weights_pretrained_filename}")

    # Calcul des distributions, KL et Surprisals
    dist_filepath = output_dir / "distributions_and_kl.json"
    kl_results, surprisal_results = evaluate_and_save_gaussian_kl(
        X_train=X_train,
        X_test=X_test,
        output_filepath=dist_filepath,
        n_bins=args.n_bins,
    )
    print(f"[INFO] Métriques de distribution enregistrées : {dist_filepath}")

    # 5. Définition et évaluation des régresseurs
    regressors = {
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
    }

    # Définition des métriques
    metrics = {"rmse": rmse, "mse": mse, "mae": mae, "r2_score": r2_score}

    # Entraînement et évaluation pour chaque modèle
    for name_model, model in regressors.items():
        print(f"\n[INFO] Entraînement du modèle : {name_model}")

        # Entraînement du modèle seul
        model.fit(X_train, y_train)

        # Création du Pipeline complet (Prétraitement + Modèle)
        full_pipeline = Pipeline([("preModel", preModel), ("regressor", model)])

        # Prédictions
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

            summary_table[metric_name] = {
                "train": perf_train,
                "test": perf_test,
                "delta": delta_perf,
                "sensitivity_gaussian": sensitivity_gaussian,
                "sensitivity_histogram": sensitivity_histogram,
            }

        # Sauvegarde locale du fichier JSON des performances
        summary_filepath = output_dir / f"{name_model}_perfs_summary.json"
        with open(summary_filepath, "w") as f:
            json.dump(summary_table, f, indent=4)
        print(f"[INFO] Résumé des performances sauvegardé : {summary_filepath}")

        # Sauvegarde locale du modèle de détection seul (ex: KNN.joblib)
        model_only_filename = output_dir / f"{name_model}.joblib"
        joblib.dump(model, model_only_filename)
        print(f"[INFO] Modèle de détection sauvegardé : {model_only_filename}")

        # Sauvegarde locale du Pipeline complet unifié (ex: KNN_full_pipeline.joblib)
        full_pipeline_filename = output_dir / f"{name_model}_full_pipeline.joblib"
        joblib.dump(full_pipeline, full_pipeline_filename)
        print(f"[INFO] Pipeline complet sauvegardé : {full_pipeline_filename}")
