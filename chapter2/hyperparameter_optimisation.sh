# This is the module for hyperparameter optimisation of the fasttext supervised learning.
# author: "joydeep bhattacharjee"<joydeepubuntu@gmail.com>

# constants
dim=(10 20)
lr=(0.1 0.3)
final=(0 0)
performance=0
i=0

# first find the installation of fasttext
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
        ./fasttext supervised -input data/dbpedia.train -output result/dbpedia -dim "$z" -lr "$y"

        # test the current model
        ./fasttext test result/dbpedia.bin data/dbpedia.test > performance.txt

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
