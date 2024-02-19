import sys
from src.load import load_signal_pc
import os


def createdir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def main(args):
    if len(args) == 1:
        position = str(args[0])

        createdir("./datas/compute")
        signals = load_signal_pc()
        with open("./datas/compute/position.csv", "a") as f:
            for bssid, signal in signals.items():
                f.write(f"{bssid};{signal};{position}\n")

    else:
        print("[USAGE]")
        print("python collect.py <position>\n")
        print("[EXEMPLE]")
        print("python collect.py 1\n")


if __name__ == "__main__":
    main(sys.argv[1:])
