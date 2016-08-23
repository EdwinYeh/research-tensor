function newM = updateM(x, m, A, seedLabel)
% x: 1*#feature vector
% m: 1*#cluster vector
% A: #cluster*#feature matrix
% seedLabel: 1*cNum seed cluster indicator
%
% return the updated m

% isSeed = 0;
% if seedLabel ~= 0
%     isSeed = 1;
% else
%     isSeed = 0;
% end

newM = m;
[cNum,d] = size(A);

%if isSeed
m(seedLabel == 1) = 1;
%end

oriLoss = calLoss(x,m,A);


% try all permutation for small #cluster
if cNum <= 4
    allPerm = perms([1:cNum]);
    minLoss = Inf;
    mOfMinLoss = zeros(1,cNum);
    
    for i=allPerm
        tempM = zeros(1,cNum);
        %if isSeed
        tempM(seedLabel==1) = 1;
        %end
        for j=1:cNum
            tempM(j) = 1;
            loss = calLoss(x,tempM,A);
            if loss <= minLoss
                minLoss = loss;
                mOfMinLoss = tempM;
            else
                break;
            end
        end
    end
    
    if oriLoss > minLoss
        newM = mOfMinLoss;
    else
        newM = m;
    end
    
else % use the dynamicM algorithm proposed in the paper for the larger #cluster
    minLoss = Inf;
    mOfMinLoss = zeros(1,cNum);
    
    for i=1:cNum % use different cluster as initialization
        tempM = zeros(1,cNum);
        %if isSeed
        tempM(seedLabel==1) = 1;
        %end
        tempM(i) = 1;
        tempMinLoss = Inf;
        tempMOfMinLoss = tempM;
        turnedOnClusterCount = length(find(tempM==1));
        for j=turnedOnClusterCount+1:cNum % run over all possible sizes (>1) of clusters turned on   
            tempOriLoss = calLoss(x,tempM,A);
            searchIndex = find(tempM~=1);
            nextM = tempM;
            for p=searchIndex % find the next cluster that will minimize the loss function
                nextM(p) = 1;
                loss = calLoss(x,nextM,A);
                if loss < tempMinLoss
                    tempMinLoss = loss;
                    tempMOfMinLoss = nextM;
                end
                nextM(p) = 0;
            end
            if tempOriLoss <= tempMinLoss % if the origin loss <= the min loss, stop to find the further more cluster to turn on
                tempMinLoss = tempOriLoss;
                tempMOfMinLoss = tempM;
                break;
            end
            tempM = tempMOfMinLoss;
        end
        
        if tempMinLoss < minLoss
            minLoss = tempMinLoss;
            mOfMinLoss = tempMOfMinLoss;
        end
    end
    
    if oriLoss > minLoss
        newM = mOfMinLoss;
    else
        newM = m;
    end
    
end



end

function l = calLoss(x,m,A)
    l = norm(x-m*A)^2;
end