PerceptionSeedRateList="0.1 0.3 0.5"

for perceptionSeedRate in $ PerceptionSeedRateList
do
    /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [21, 154], $perceptionSeedRate)" &
done

for perceptionSeedRate in $ PerceptionSeedRateList
do
    /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [21, 154, 131], $perceptionSeedRate)" &
done
