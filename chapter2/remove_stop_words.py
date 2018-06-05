"""
Small script so that it is easy to work with the pipe operator and can be
plugged in easily with other bash commands.

Usage: cat raw_data.txt | python remove_stop_words.py > no_stop_words.txt

"""
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys


def get_lines():
    """Process lines from standard input.

    :yields: str: each line.

    """
    lines = sys.stdin.readlines()
    for line in lines:
        yield line


def main():
    """Split the line, remove the stop words, join and serve."""
    stop_words = set(stopwords.words('english'))
    for line in get_lines():
        words = line.lower().split()
        newwords = [w for w in words if w not in stop_words]
        print(' '.join(newwords))


if __name__ == "__main__":
    main()
