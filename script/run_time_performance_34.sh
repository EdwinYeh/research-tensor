
Idx_dataset="1"

for idx in $Idx_dataset
do
    /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; datasetId = $idx; sampleSizeLevel=''; Exp_DY_cross_domain; Exp_DY_cross_domain_3way; sampleSizeLevel='bigSample/'; Exp_DY_cross_domain; Exp_DY_cross_domain_3way;  exit;" &
done
