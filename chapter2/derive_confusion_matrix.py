# __author__ = "joydeep bhattacharjee"
# The commands in this file are based on transformations.sh and assume the same
# file structure. For your custom implementation make those required changes.

# Extract the labels
cut -f 1 -d ' ' data/yelp/val.txt > data/yelp/val.testlabel

# Extract the sentences
cut -f 2- -d ' ' data/yelp/val.txt > data/yelp/val.testsentences

# Put the predictions to a file
fasttext predict result/yelp/star_model.bin data/yelp/val.testsentences > pexp

# Run the confusion matrix
python fasttext_confusion_matrix.py data/yelp/val.testlabel pexp

# Delete the helper files
rm pexp data/yelp/val.testlabel data/yelp/val.testsentences
