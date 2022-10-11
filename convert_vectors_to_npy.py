"""
Сохраняет готовые эмбеддинги для языка пушту в папку data.
Источник эмбеддингов – https://github.com/Junaid199f/Pashto-POS-Tagging-Project
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import find_encoding

(path := Path("data")).mkdir(exist_ok=True)


def main() -> None:
    """
    Читает txt-файл с эмбеддингами и сохраняет их в файл GloVe.npz в папку data
    """
    embeddings = dict()

    with open("vectors.txt", encoding=find_encoding("vectors.txt")) as file:
        for line in tqdm(file.readlines()):
            values = line.split()
            embeddings[values[0]] = np.asarray(values[1:], dtype="float32")

    np.savez(path / "GloVe.npz", **embeddings)


if __name__ == "__main__":
    main()
