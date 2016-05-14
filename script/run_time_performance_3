rm -r ../../exp_result/DY_corss/
Idx_dataset="1 5"

for idx in $Idx_dataset
do
    /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; datasetId = $idx; Exp_DY_cross_domain; exit;" &
done
