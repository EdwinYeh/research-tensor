
cluster_userIdx = cluster_userIdx + 1; % from 0~99 to 1~100;
userCount = cluster_userIdx(end);
startFromUser = 1;

sigmas = [30,50,70,90];
alphas = [0.1,1,10];
betas = [0.1,1,10];


iter = 1;
load([parentPath inputPath datasetName 'Seed.mat'],'seedDataMat');
[~,seedSetNum] = size(seedDataMat);
userDataMat = cell(userCount,seedSetNum);% used to store result cells of users returned by gridSearch
[d,n] = size(X);

for u=startFromUser:userCount
    u
    disp(['NPE start from user ' num2str(u)]);
    clusterIdx = find(cluster_userIdx == u);
    trueC = cluster_data(clusterIdx,:);
    trueC = trueC'; % transport to n*k
    P = allP(:,clusterIdx); 
    P = P(find(sum(P,2)~=0),:);
    
    [~,k] = size(trueC); % #cluster
    [d,n] = size(X);
    [p,~] = size(P);

    for seedSetCount=1:seedSetNum
        % pick 2 seeds from each cluster
        S = seedDataMat{u,seedSetCount};
        S2 = zeros(1,n);
        P2 = zeros(n,p);
        for ss=1:k
            seedIdx = find(S(:,ss)==1);
            S(seedIdx,ss) = 1;
            S2(seedIdx) = 1;
            P2(seedIdx,:) = P2(seedIdx,:) + [P(:,ss),P(:,ss)]';
        end
        S2 = diag(S2);
        
        for i=1:iter
            try
            [curUserResult] = gridSearch4(X, P2, S2, trueC, sigmas, alphas, betas); % [sigma, alpha, beta, FScore, userResult]

            userDataMat{u,seedSetCount} = curUserResult;
            save([outputPath datasetName num2str(startFromUser) '-' num2str(userCount) '_NPE_result.mat'], 'userDataMat', 'seedDataMat', 'sigmas' ,'alphas', 'betas');
            catch exception
                disp(exception);
                continue;
            end
        end
    
    end
end
