UserIdList="3 5 7 6 9"

for userId in $UserIdList
do
 /usr/local/MATLAB/R2012a/bin/matlab -r "cd ..; Exp_clustering('citation', [0, $userId])" &
done
