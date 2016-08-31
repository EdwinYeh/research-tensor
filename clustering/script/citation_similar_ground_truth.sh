UserIdList="37 44 11 17 97"

for userId in $UserIdList
do
 /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [38, $userId])" &
 done
