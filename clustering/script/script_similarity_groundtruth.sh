UserIdList="75 79 85 90 65"

for userId in $UserIdList
do
	/usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('mturk', [0, userId]);"
done