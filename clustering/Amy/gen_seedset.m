function [] = gen_seedset(X, cluster_data,cluster_userIdx,parentPath,inputPath,datasetName,userIdList,numSeedSet)
    cluster_userIdx = cluster_userIdx + 1;
    userCount = length(userIdList);
    seedDataMat = cell(userCount,100);
    
    for userId=userIdList
        clusterIdx = find(cluster_userIdx == userId);
        trueC = cluster_data(clusterIdx,:);
        trueC = trueC'; % transport to n*k
        [~,k] = size(trueC); % #cluster
        [~,n] = size(X);
        for seedSetCount=1:numSeedSet
            S = zeros(n,k);
            for ss=1:k
                memberIdx = find(trueC(:,ss)==1);
                seedIdx = memberIdx(randperm(length(memberIdx),2));
                S(seedIdx,ss) = 1;
            end
            seedDataMat{userId,seedSetCount} = S;
        end
    end
    load
    save([parentPath inputPath 'Seed.mat'],'seedDataMat');
end