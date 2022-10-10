"""
POS-тэггер языка пушту. Проходится по всем текстам в папке texts, размечает их
по частям речи и сохраняет размеченные тексты в папку tagged_texts.
Использует модель из https://github.com/Junaid199f/Pashto-POS-Tagging-Project
"""

from argparse import ArgumentParser
from string import punctuation
from typing import Generator, List

import pickle
from pathlib import Path

import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from tqdm import tqdm

punctuation += "؟،"

texts_path = Path("texts")
(tagged_texts_path := Path("tagged_texts")).mkdir(exist_ok=True)

with open("data.pkl", "rb") as f:
    X_train, Y_train, word2int, int2word, tag2int, int2tag = pickle.load(f)
    del X_train
    del Y_train

model = load_model("model.h5")


def split_line(line: str) -> Generator[List[str], None, None]:
    """
    Разбивает строку на куски длиной в 50 или меньше слов. Это нужно, потому
    что модель принимает на вход только последовательности длиной 50.

    Args:
        line: a line to be splitted

    Returns:
        a list of sequences, each sequence having no more than 50 words
    """
    words = [word.strip(punctuation) for word in line.split()]
    words = [word for word in words if word]

    for i in range(0, len(words), 50):
        yield words[i : i + 50]


def tag_sentence(sentence: List[str], verbose: int) -> str:
    """
    Размечает предложение по частям речи.

    Args:
        sentence: a list of words to be tagged
        verbose: verbosity level

    Returns:
        tokenized and tagged sentence (words are split by line break)
    """
    if verbose > 1:
        print("Tagging sentence: " + "".join(sentence))

    padded_tokenized_sentence = pad_sequences(
        np.asarray(
            [[word2int[word] for word in sentence if word in word2int]]
        ),
        maxlen=50,
    )

    prediction = model.predict(padded_tokenized_sentence)
    result = []

    for i, word in enumerate(sentence):
        pred = int2tag[np.argmax(prediction[0][i][1:]) + 1]
        result.append(pred + "/" + word)

    if verbose > 2:
        for elem in result:
            print(elem)

    return "\n".join(result)


def tag_text(text: Path, verbose: int) -> None:
    """
    Размечает текст по частям речи и сохраняет результат в папку tagged_texts.

    Args:
        text: path to text
        verbose: verbosity level
    """
    with text.open() as file:
        lines = file.read().split("\n")

    result = []

    if verbose:
        for line in tqdm(lines, desc=f"Tagging file {text.name}..."):
            for sentence in split_line(line):
                result.append(tag_sentence(sentence, verbose))
    else:
        for line in lines:
            for sentence in split_line(line):
                result.append(tag_sentence(sentence, verbose))

    with (tagged_texts_path / text.name).open("w") as file:
        file.write("\n".join(result))


def main(verbose: int) -> None:
    """
    Проходится по всем текстам в папке texts, размечает их по частям речи и
    сохраняет размеченные тексты в папку tagged_texts.

    Args:
        verbose: verbosity level
    """
    for text in tqdm(sorted(texts_path.iterdir()), desc="Tagging texts..."):
        tag_text(text, verbose)


if __name__ == "__main__":
    parser = ArgumentParser(prog="pos_tagger", description="Pashto POS tagger")

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity level"
    )

    args = parser.parse_args()
    main(args.verbose)
