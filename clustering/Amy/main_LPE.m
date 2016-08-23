
cluster_userIdx = cluster_userIdx + 1; % from 0~99 to 1~100;
userCount = length(userIdList);

alphas = [1, 10];
betas = [0.1, 1];


iter = 1;
[~,seedSetNum] = size(seedDataMat);
userDataMat = cell(userCount,seedSetNum);% used to store result cells of users returned by gridSearch
[d,n] = size(X);

for userId=userIdList
    userId
    clusterIdx = find(cluster_userIdx == userId);
    trueC = cluster_data(clusterIdx,:);
    trueC = trueC'; % transport to n*k
    P = allP(:,clusterIdx); 
    P = P(find(sum(P,2)~=0),:);
    
    [~,k] = size(trueC); % #cluster
    [d,n] = size(X);
    [p,~] = size(P);

    for seedSetCount=1:seedSetNum
        % pick 2 seeds from each cluster
        SeedCluster = seedDataMat{userId,seedSetCount};
        S2 = zeros(1,n);
        P2 = zeros(n,p);
        for clusterId=1:k
            seedIdx = find(SeedCluster(:,clusterId)==1);
            SeedCluster(seedIdx,clusterId) = 1;
            S2(seedIdx) = 1;
            P2(seedIdx,:) = P2(seedIdx,:) + [P(:,clusterId),P(:,clusterId)]';
        end
        S2 = diag(S2);
        
        for i=1:iter
            try
            [curUserResult] = gridSearch_LPE(X, P2, S2, alphas, betas);

            userDataMat{userId,seedSetCount} = curUserResult;
            save([outputPath datasetName num2str(userId) '_LPE_result.mat'], 'userDataMat', 'seedDataMat','alphas', 'betas');
            catch exception
                disp(exception);
                continue;
            end
        end
    
    end
end
