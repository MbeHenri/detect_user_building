import joblib
import mlflow
from numpy import mean
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.load import load_collected_positions_2d
from src.perfs import rmse, mse, mae

# Configuration de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Positionnement_2D_Regressors")

# Récuperation des positions et des signaux collectés
P, D = load_collected_positions_2d("datas/without_acces/BSSIDs_collected.csv")

preModel = Pipeline(
    [("DicVect", DictVectorizer(sparse=False)), ("Scaler", StandardScaler())]
)
datas = preModel.fit_transform(D)


# Définition des modèles de régression
model = DecisionTreeRegressor()


# Définition des fonctions d'évaluation
metrics = {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2_score}

kf = KFold(n_splits=10, shuffle=True, random_state=42)
perfs = {name_metric: [] for name_metric in metrics.keys()}

# 1. Enregistrer le Pipeline de prétraitement global comme un artefact partagé
with mlflow.start_run(run_name="Pre-processing_Pipeline"):
    weights_pretrained_filename = "parameters/premodel_pipeline.joblib"
    joblib.dump(preModel, weights_pretrained_filename)
    mlflow.log_artifact(weights_pretrained_filename)

# Entraînement et évaluation des modèles avec validation croisée
# LANCEMENT GLOBALE
with mlflow.start_run(run_name=f"Global_arbre_decision") as parent_run:
    mlflow.log_param("model_type", "arbre_decision")

    # Boucle de validation croisée
    for fold, (train_index, test_index) in enumerate(kf.split(datas)):
        X_train, X_test = datas[train_index], datas[test_index]
        y_train, y_test = P[train_index], P[test_index]

        # RUN SPECIFIQUE A UNE SELECTION (TRAIN - TEST)
        with mlflow.start_run(run_name=f"arbre_decision_Fold_{fold}", nested=True):
            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calcul et enregistrement des métriques
            for name_metric in metrics.keys():
                metric_value = metrics[name_metric](y_test, y_pred)
                perfs[name_metric].append(metric_value)
                mlflow.log_metric(name_metric, metric_value)

            # SAUVEGARDE DES POIDS : Export du modèle de ce pli spécifique en .joblib
            weight_filename = f"parameters/arbre_decision_fold_{fold}.joblib"
            joblib.dump(model, weight_filename)

            # Téléversement du fichier de poids comme artefact de ce pli dans MLflow
            mlflow.log_artifact(weight_filename, artifact_path="model_weights")

    # Calcul et enregistrement des performances moyennes globales dans le RUN GLOBAL
    for name_metric in metrics.keys():
        mean_metric_value = mean(perfs[name_metric])
        mlflow.log_metric(f"mean_{name_metric}", mean_metric_value)
