"""
A small script that will print the confusion matrix.

Usage
------------
You will need to have the test files and the label files in separate files to
be able to use this script. Lets say you have all the sentences in the
test_sentences.txt file and the corresponding labels in the test_labels.txt
file. You can then run fasttext predict on the sentences to get the
corresponding predicted labels. Store them in the pexp file.

    $ ./fasttext predict model.bin test_sentences.txt > pexp

So now you can run these two files through the script and it should give you
the confusion matrix.

    $ python fasttext_confusion_matrix.py test_labels.txt pexp

"""
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix


def parse_labels(path):
    with open(path, 'r') as f:
        return np.array(list(map(lambda x: x[9:], f.read().split())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display confusion matrix.')
    parser.add_argument('test', help='Path to test labels')
    parser.add_argument('predict', help='Path to predictions')
    args = parser.parse_args()
    test_labels = parse_labels(args.test)
    pred_labels = parse_labels(args.predict)
    eq = test_labels == pred_labels
    print("Accuracy: " + str(eq.sum() / len(test_labels)))
    print(confusion_matrix(test_labels, pred_labels))
