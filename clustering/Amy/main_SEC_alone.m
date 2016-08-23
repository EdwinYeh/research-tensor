cluster_userIdx = cluster_userIdx + 1; % 0~99 to 1~100
trueC = cluster_data;%groundTrue_cluster_data;

mu = [0.01,1,100];
gama = [0.0001,0.01,1,100,10000];
u = max(cluster_userIdx);
fromUser = 1;
toUser = u;



[d,n] = size(X);
load([parentPath inputPath datasetName 'Seed.mat'],'seedDataMat');
[~,seedSetNum] = size(seedDataMat);
recall = zeros(u,length(mu),length(gama),seedSetNum);
filterPrecision = zeros(u,length(mu),length(gama),seedSetNum);
FScore = zeros(u,length(mu),length(gama),seedSetNum);
recallsOfBestFScore = zeros(u,1);
precisionsOfBestFScore = zeros(u,1);
bestFScore = zeros(u,1);
bestFScore = bestFScore - inf;

paramNameOrder = 'mu, gama';
bestParameter = cell(u,1);

clusteringResult = cell(u,length(mu),length(gama),seedSetNum);
initIterNum = 2;

fileParamName = 'mu';
for i=mu
    fileParamName = [fileParamName num2str(i) '-'];
end
fileParamName = [fileParamName 'gama'];
for i=gama
    fileParamName = [fileParamName num2str(i) '-'];
end

for i=fromUser:toUser
    disp(['SEC alone start from user: ' num2str(fromUser)]);
    i
    
    k = length(find(cluster_userIdx==i));
    userTrueC = trueC(find(cluster_userIndex(:,1) == i-1),:);
    
    %augment perception features to seeds
    clusterIdx = find(cluster_userIdx == i);
    P = allP(:,clusterIdx);
    P = P(find(sum(P,2)~=0),:);
    
    for seedSetCount=1:seedSetNum
    
        augmentFeature = zeros(length(P(:,1)),n);

        % pick 2 seeds from each cluster
        S = seedDataMat{i,seedSetCount};%zeros(n,k);
        for ss=1:k
            seedIdx = find(S(:,ss)==1);
            augmentFeature(:,seedIdx) = augmentFeature(:,seedIdx) + [P(:,ss),P(:,ss)];
        end
        augmentX = [X;augmentFeature];
        

        % SEC
        for muCount=1:length(mu)
            for gamaCount=1:length(gama)
                highestFScore = 0;
                for initIdx=1:initIterNum
                    try
                        newF = SEC(augmentX,k, mu(muCount), gama(gamaCount)); % eat d*n, return n*k
                        %seedOKM
                        clustering = seedOKM(newF,k,S',0.0001); %return k*n
                        result = clustering'; %transform to n*k

                    catch exception
                        disp(['error at user ', num2str(i)]);
                        disp(exception);
                        continue;
                    end

                    % calculate the recall and precision
                    base = sum(sum(userTrueC'));
                    intersect = userTrueC' & result;
                    theRecall = sum(sum(intersect))/base;
                    theFilterPrecision = filteredPrecision(userTrueC', result);
                    TheFScore = 2*((theRecall*theFilterPrecision)/(theRecall+theFilterPrecision))
                    
                    
                    if(highestFScore < TheFScore)
                        highestFScore = TheFScore;
                        clusteringResult{i,muCount,gamaCount,seedSetCount} = result;
                        
                        recall(i,muCount,gamaCount,seedSetCount) = theRecall;
                        filterPrecision(i,muCount,gamaCount,seedSetCount) = theFilterPrecision;
                        FScore(i,muCount,gamaCount,seedSetCount) = TheFScore;
                    end
                end
            end
        end
    end
    for muCount=1:length(mu)
        for gamaCount=1:length(gama)
            avgF = mean(FScore(i,muCount,gamaCount,:));
            if(bestFScore(i,1) < avgF)
                bestFScore(i,1) = avgF;
				recallsOfBestFScore(i,1) = mean(recall(i,muCount,gamaCount,:));
				precisionsOfBestFScore(i,1) = mean(filterPrecision(i,muCount,gamaCount,:));
                bestParameter{i,1} = [muCount,gamaCount];
            end
        end
    end
    
    save([outputPath datasetName num2str(fromUser) '-' num2str(toUser) '_SEC_alone_' fileParamName '_with_SeedOKM_recall_precision.mat'], 'recallsOfBestFScore','precisionsOfBestFScore', 'recall','filterPrecision','FScore','bestFScore','bestParameter','paramNameOrder','clusteringResult');
end

avgR = mean(recallsOfBestFScore(find(recallsOfBestFScore>0)));
avgP = mean(precisionsOfBestFScore(find(precisionsOfBestFScore>0)));

disp(['recall: ' num2str(avgR)]);
disp(['precision: ' num2str(avgP)]);
disp(['FScore: ' num2str(2*((avgR*avgP)/(avgR+avgP)))]);

