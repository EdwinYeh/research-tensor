PerceptionSeedRateList="0.3 0.5 0.7"

for perceptionSeedRate in $PerceptionSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0], $perceptionSeedRate, 1)" &
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('song', [0], $perceptionSeedRate, 1)" &
done
