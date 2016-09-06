UserIdList="154 131 111 13 3"

for userId in $UserIdList
do
 /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [21, $userId])" &
 done
