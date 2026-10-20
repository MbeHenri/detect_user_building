from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# Définition des fonctions d'évaluation
def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def mse(y_true, y_pred):
    # Calcul manuel du MSE car root_mean_squared_error donne le RMSE directement
    return mean_absolute_error(
        y_true, y_pred
    )  # Ou utilisez mean_squared_error importé de sklearn


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
