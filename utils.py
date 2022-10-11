"""
Вспомогательные функции.
"""

import chardet


def find_encoding(filename: str) -> str:
    """
    Определяет кодировку файла.

    Args:
        filename: path to a file

    Returns:
        file encoding
    """
    with open(filename, "rb") as file:
        chars = file.read()

    return chardet.detect(chars)["encoding"]
