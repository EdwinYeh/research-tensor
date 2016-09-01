UserIdList="0 6 7 1 2"

for userId in $UserIdList
do
 /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [3, $userId])" &
done
