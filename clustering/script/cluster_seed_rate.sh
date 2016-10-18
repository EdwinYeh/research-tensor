ClusterSeedRateList="0.25 0.5 0.75"

for clusterSeedRate in $ClusterSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0, 64, 74], 0.5, $clusterSeedRate);" &
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('song', [0, 3, 5], 0.5, $clusterSeedRate);" &
done
