Idx_dataset="1 2 3 4 5 6 7 8 9"

for idx in $Idx_dataset
do
    /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; datasetId = $idx; Exp_DY; exit;" &
done
