import json
import fastText
from fastText import load_model
import os
from flask import Flask
from flask import g
from flask import jsonify
from flask import request

from ft_utils import nn
from ft_utils import FT_MODEL

app = Flask(__name__)


@app.route('/')
def hello():
    return 'This is a fasttext server'


@app.route('/nn/<question_word>')
def nearest_neighbours(question_word):
    """TODO: Docstring for nearest_neighbours.

    :question_word: TODO
    :returns: TODO

    """
    answers = [a for a in nn(FT_MODEL, question_word, k=5)]
    return jsonify(dict(question=question_word, answers=answers))


if __name__ == "__main__":
    app.run()
