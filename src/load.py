from numpy import array
from pandas import read_csv


def load_acces_points_2d(path_csv):
    df = read_csv(path_csv, sep=";")
    A = df[["X", "Y"]].values
    W = df["Signal(%)"].values
    return A, W


def load_to_compute_acces_points_2d(path_csv):
    df = read_csv(path_csv, sep=";")
    C = {}
    for bssid, signaux in df.groupby("BSSID"):
        C[bssid] = {}
        C[bssid]["A"] = signaux[["X", "Y"]].values
        C[bssid]["W"] = signaux["Signal(%)"].values
    return C


def load_acces_points_save_2d(path_csv):
    df = read_csv(path_csv, sep=";")
    acces_points = {}
    for i in range(df.shape[0]):
        acces_points[df["BSSID"][i]] = [df["X"][i], df["Y"][i]]
    return acces_points


def load_collected_positions_2d(path_csv):
    df = read_csv(path_csv, sep=";")
    P = []
    D = []
    for position, signaux in df.groupby(["X", "Y"]):
        P.append(list(position))
        signaux_ = (
            signaux[["BSSID", "Signal(%)"]].set_index("BSSID").to_dict()["Signal(%)"]
        )
        D.append(signaux_)
    P = array(P, dtype=float)
    return P, D


def load_collected_user(path_csv):
    df = read_csv(path_csv, sep=";")
    E = df.set_index("BSSID").to_dict()["Signal(%)"]
    return E
