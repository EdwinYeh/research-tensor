cluster_userIdx = cluster_userIdx + 1; % 0~99 to 1~100
trueC = cluster_data;


u = max(cluster_userIdx);
fromUser = 1;
toUser = u;


[d,n] = size(X);
load([parentPath inputPath datasetName 'Seed.mat'],'seedDataMat');
[~,seedSetNum] = size(seedDataMat);
recall = zeros(u,seedSetNum);
filterPrecision = zeros(u,seedSetNum);
FScore = zeros(u,seedSetNum);
clusteringResult = cell(u,seedSetNum);
initIterNum = 2;

for i=fromUser:toUser
    disp(['CCSKL alone start from user: ' num2str(fromUser)]);
    i
    
    k = length(find(cluster_userIdx==i));
    userTrueC = trueC(find(cluster_userIndex(:,1) == i-1),:);
    
    
    clusterIdx = find(cluster_userIdx == i);
    P = allP(:,clusterIdx);
    P = P(find(sum(P,2)~=0),:);
    
    for seedSetCount=1:seedSetNum
        %augment perception features to seeds
        augmentFeature = zeros(length(P(:,1)),n);

        % pick 2 seeds from each cluster
        S = seedDataMat{i,seedSetCount};%zeros(n,k);
        for ss=1:k
            seedIdx = find(S(:,ss)==1);
            augmentFeature(:,seedIdx) = augmentFeature(:,seedIdx) + [P(:,ss),P(:,ss)];
        end
        augmentX = [X;augmentFeature];

        S2 = S'; % k*n

        %process must-link (#must-link * n)
        S2 = S2(find(sum(S2,2) ~= 0),:);

        % CCSKL
        highestFScore = 0;
        for initIdx=1:initIterNum
            try
                dimReduction = 20;
                newF = CCSKL(augmentX, k, S2, [], dimReduction); %return n*k

                %OKM
                clustering = seedOKM(newF,k,S',0.0001);%return k*n
                result = clustering'; %transform to n*k
            

            % calculate the recall and precision
            base = sum(sum(userTrueC'));
            intersect = userTrueC' & result;
            theRecall = sum(sum(intersect))/base;
            theFilterPrecision = filteredPrecision(userTrueC', result);
            TheFScore = 2*((theRecall*theFilterPrecision)/(theRecall+theFilterPrecision))
            
            if(highestFScore < TheFScore)
                highestFScore = TheFScore;
                clusteringResult{i,seedSetCount} = result;
                
                recall(i,seedSetCount) = theRecall;
                filterPrecision(i,seedSetCount) = theFilterPrecision;
                FScore(i,seedSetCount) = TheFScore;
            end
            
            catch exception
                disp(['error at user ', num2str(i)]);
                disp(exception);
                continue;
            end
        end
        

        %save([outputPath datasetName num2str(fromUser) '-' num2str(toUser) '_CCSKL_alone_m' num2str(dimReduction) '_recall_precision.mat'], 'recall','filterPrecision','FScore','clusteringResult');

    end
end

avgRs = zeros(u,1);
avgPs = zeros(u,1);
for i=fromUser:toUser
    avgRs(i) = mean(recall(i,(find(recall(i,:)>0))));
    avgPs(i) = mean(filterPrecision(i,(find(filterPrecision(i,:)>0))));
end
avgR = mean(avgRs(find(avgRs>0)));
avgP = mean(avgPs(find(avgPs>0)));

disp(['recall: ' num2str(avgR)]);
disp(['precision: ' num2str(avgP)]);
disp(['FScore: ' num2str(2*((avgR*avgP)/(avgR+avgP)))]);

