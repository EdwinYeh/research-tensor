Idx_dataset="7 8 9 10 11 12 13"

for idx in $Idx_dataset
do
    /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; datasetId = $idx; main_DX_new_project; exit;" &
done
