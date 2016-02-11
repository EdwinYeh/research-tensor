Idx_dataset="1 2 3 4 5 6 7 8 9 10 11 12 13"
for idx in $Idx_dataSet
do
	/usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; datasetId=$idx; Exp_ours_allfree_cvx_sigma; exit;" &
done