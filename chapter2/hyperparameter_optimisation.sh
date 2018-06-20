dim=(10 20)
lr=(0.1 0.3)
epochs=(5 10)

final=(0 0 0)
performance=0

for z in ${dim[@]}
do
    for y in ${lr[@]}
    do
        for x in ${epochs[@]}
        do
            # train with the current set of parameters
            ./fasttext supervised -input train.txt -output model -dim "$z" -lr "$y" -epoch "$x"
            
            # test the current model
            ./fasttext test model.bin test.txt > performance.txt
            
            # see if current model is the best model and update the performance variable.
            present_performance=$(cat performance.txt | awk 'NR==2 {print $2}') # get the precision values
            if (( $(echo "$present_performance > $performance" | bc -l) )); then
                # if current performance is the best performance till date
                final[0]="$z"
                final[1]="$y"
                final[2]="$x"
                echo "Performance values changed to ${final[@]}"
                echo "present accuracy:"
                cat performance.txt
            fi
            
        done
    done
done
