from pandas import read_csv
from src.detect_user import detect_user_position
import sys

def load_acces_points(path_csv):
    df = read_csv(path_csv, sep=";")
    A = df[["X", "Y"]].values
    W = df["Signal(%)"].values
    return A, W

def dist(weight, a= -50, n=4):
    return pow(10, (a - weight)/(10 * n))

A, W = load_acces_points("./BSSID.csv")
M, _ = detect_user_position(A, W, d=dist)

print("position détecté")
print(f"x : {M[0]}, y : {M[1]}")

