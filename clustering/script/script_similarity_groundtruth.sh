UserIdList="74 78 84 89 64"

for userId in $UserIdList
do
	/usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0, $userId]);" &
done
