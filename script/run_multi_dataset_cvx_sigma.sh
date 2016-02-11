Idx_dataSet="3 5 6 9 10"

for idx in $Idx_dataSet
do
	/usr/local/MATLAB/R2012a/bin/matlab -r "datasetId=$idx; cd ..; Exp_ours_allfree_cvx_sigma; exit;" &
done
