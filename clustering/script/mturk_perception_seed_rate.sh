PerceptionSeedRateList="0.1 0.3 0.5"

for perceptionSeedRate in $PerceptionSeedRateList
do
 /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0, 64], perceptionSeedRate)" &
done

for perceptionSeedRate in $PerceptionSeedRateList
do
 /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0, 64, 74], perceptionSeedRate)" &
done
