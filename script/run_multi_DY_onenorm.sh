Idx_dataset="1 2 3 4 5 6 7 8 9 10 11 12 13"

for Idx in $Idx_dataset
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd..; datasetId = $Idx; Exp_DY_onenorm; exit;" &
done
