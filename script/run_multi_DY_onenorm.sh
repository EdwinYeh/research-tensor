Idx_dataset="1 2 3 7 10"

for idx in $Idx_dataset
do
    /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; datasetId = $idx; Exp_DY_onenorm; exit;" &
done
