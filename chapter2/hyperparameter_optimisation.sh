#!/bin/bash
# This is the module for hyperparameter optimisation of the fasttext supervised learning.
# author: "joydeep bhattacharjee"<joydeepubuntu@gmail.com>

###########################################################################################
###########################  read options ##################################
###########################################################################################
usage() { echo "Usage: $0 -l <training file> -t <test file>" 1>&2; exit 1; }

while getopts ":l:t:" o; do
    case "${o}" in
        l)
            training_file=${OPTARG}
            ;;
        t)
            testing_file=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${training_file}" ] && [ -z "${testing_file}" ]; then
	usage
else
	if ! [ -s "$training_file" ]; then
		echo "Training file is either not correct path or is empty."
		usage
	fi
	if ! [ -s "$testing_file" ]; then
		echo "Testing file is either not correct path or is empty."
		usage
	fi
	echo "Training file found: $training_file"
	echo "Testing file found: $testing_file"
fi

###############################################################

# constants
dim=(10 20)
lr=(0.1 0.3)
final=(0 0)
performance=0
i=0

# first find the installation of fasttext
fasttext="$(command -v fasttext)"
if ! [ -x "$(command -v fasttext)" ]; then
  echo 'fasttext is not installed globally. Lets try a local build'
  if [ -f fasttext ]; then
      fasttext=./fasttext
  else
      echo 'Error: Could not find fasttext.' >&2
      exit 1
  fi
fi

# make the comparisons
for z in ${dim[@]}
do
    for y in ${lr[@]}
    do
        # train with the current set of parameters
        "$fasttext" supervised -input "$training_file" -output _hyper_parameter_tmp_model -dim "$z" -lr "$y"

        # test the current model
        "$fasttext" test _hyper_parameter_tmp_model.bin "$testing_file" > performance.txt

        # selecting the best performance
        present_performance=$(cat performance.txt | awk 'NR==2 {print $2}')
        if (( $(echo "$present_performance > $performance" | bc -l) )); then
            final[0]="$z"
            final[1]="$y"
            echo "Performance values changed to ${final[@]}"
            echo "Present precision recall values are:"
            cat performance.txt
        fi
    done
done

echo "Publish the final results"
echo "the final model parameters are:"
echo "dim: ${final[0]}"
echo "lr: ${final[1]}"

# clean up
rm _hyper_parameter_tmp_model*  performance.txt
