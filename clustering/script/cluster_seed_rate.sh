ClusterSeedRateList="0.3 0.6 0.9"

for clusterSeedRate in $ClusterSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0, 64, 74], 0.5, $clusterSeedRate);" &
done
