PerceptionSeedRateList="0.5 0.7"

for perceptionSeedRate in $PerceptionSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [22], $perceptionSeedRate, 1)" &
done

for perceptionSeedRate in $PerceptionSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [22,83], $perceptionSeedRate, 1)" &
done

for perceptionSeedRate in $PerceptionSeedRateList
do
  /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [22,83,144], $perceptionSeedRate, 1)" &
done
