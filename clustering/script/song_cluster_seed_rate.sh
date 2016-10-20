ClusterSeedRateList="0.33 0.66 1"

for clusterSeedRate in $ClusterSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('song', [0, 3, 5], 0.5, $clusterSeedRate);" &
done
