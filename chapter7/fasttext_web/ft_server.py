"""
This module has the relevant flask endpoints.
"""
import os
from flask import Flask
from flask import jsonify

from ft_utils import nn
from ft_utils import FT_MODEL

app = Flask(__name__)


@app.route('/')
def hello():
    return 'This is a fasttext server'


@app.route('/nn/<question_word>')
def nearest_neighbours(question_word):
    """Pass the question word to the nearest neighbor function and serve the
    answers.

    :question_word: str

    """
    answers = [a for a in nn(FT_MODEL, question_word, k=5)]
    return jsonify(dict(question=question_word, answers=answers))


if __name__ == "__main__":
    app.run()
