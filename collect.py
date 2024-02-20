import sys
from src.load import load_signal_pc
import os
import json


def createdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load2json(path) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            extensions_images = json.load(f)
        return extensions_images
    return {}


def save2json(obj, path_file):
    with open(path_file, "w") as fichier_json:
        json.dump(obj, fichier_json)


def main(args):
    if len(args) == 1:
        position = str(args[0])

        createdir("./datas/compute")
        controller = load2json("./datas/compute/controller.json")

        try:
            nbexec = controller[position] + 1
        except Exception:
            nbexec = 1

        signals = load_signal_pc()
        with open(f"./datas/compute/datei_{position}.csv", "a") as f:
            for bssid, signal in signals.items():
                f.write(f"{bssid};{signal};{nbexec}\n")

        controller[position] = nbexec
        save2json(controller, "./datas/compute/controller.json")

    else:
        print("[USAGE]")
        print("python collect.py <position>\n")
        print("[EXEMPLE]")
        print("python collect.py 1\n")


if __name__ == "__main__":
    main(sys.argv[1:])
