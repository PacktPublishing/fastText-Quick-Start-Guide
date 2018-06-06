# Extract the label and the text only.
cat data/yelp/yelp_review.csv | python parse_yelp_dataset.py > data/yelp/yelp_review.v1.csv

# remove the stop words from the text
cat data/yelp/yelp_review.v1.csv | python remove_stop_words.py > data/yelp/yelp_review.v2.csv

# convert all upper case to lower case
cat data/yelp/yelp_review.v2.csv | tr '[:upper:]' '[:lower:]' > data/yelp/yelp_review.v3.csv

# append the labels with the label keyword
cat data/yelp/yelp_review.v3.csv| sed -e 's/^/__label__/g' > data/yelp/yelp_review.v4.csv

# normalise some of the punctuation
cat data/yelp/yelp_review.v4.csv | sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' > data/yelp/yelp_review.v5.csv
cat data/yelp/yelp_review.v5.csv | sed 's/\,//g' > data/yelp/yelp_review.v6.csv
cat data/yelp/yelp_review.v6.csv | sed 's/\.//g' > data/yelp/yelp_review.v7.csv

# trim excess whitespace
cat data/yelp/yelp_review.v7.csv| tr -s " " > data/yelp/yelp_review.v8.csv

# shuffle the dataset
perl -MList::Util -e 'print List::Util::shuffle <>' data/yelp/yelp_review.v8.csv > data/yelp/yelp_review.v9.csv

# split into training and validation
awk -v lines=$(wc -l < data/yelp/yelp_review.v9.csv) -v fact=0.80 'NR <= lines * fact {print > "traintxt"; next} {print > "valtxt"}' data/yelp/yelp_review.v9.csv

# move into respective folders
mv traintxt data/yelp/train.txt
mv valtxt data/yelp/val.txt
