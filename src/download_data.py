"""Download C-MAPSS files (public GitHub mirror of the NASA release)."""
import urllib.request
from src.config import DATA_DIR

BASE = "https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/master/C-MAPSS-Data"


def main():
    for i in range(1, 5):
        for p in ("train", "test", "RUL"):
            name = f"{p}_FD00{i}.txt"
            dst = DATA_DIR / name
            if dst.exists():
                continue
            print("downloading", name)
            urllib.request.urlretrieve(f"{BASE}/{name}", dst)
    print("done ->", DATA_DIR)


if __name__ == "__main__":
    main()
