PerceptionSeedRateList="0.3 0.5 0.7"

for perceptionSeedRate in $PerceptionSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0,64], $perceptionSeedRate, 1)" &
done

for perceptionSeedRate in $PerceptionSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0,64,74], $perceptionSeedRate, 1)" &
done
