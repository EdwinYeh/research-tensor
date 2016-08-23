
algoResultMat = '';
load([outputPath algoResultMat '.mat']);
load([parentPath inputPath datasetName '.mat']);
trueC = cluster_data;

mu = [0.01,1,100];
gama = [0.0001,0.01,1,100,10000];
u = length(userDataMat);
fromUser = 1;
toUser = u;


[~,seedSetNum] = size(seedDataMat);
[diffParamNum,~] = size(userDataMat{1,1});
recall = zeros(u,diffParamNum,length(mu),length(gama),seedSetNum);
filterPrecision = zeros(u,diffParamNum,length(mu),length(gama),seedSetNum);
FScore = zeros(u,diffParamNum,length(mu),length(gama),seedSetNum);
recallsOfBestFScore = zeros(u,1);
precisionsOfBestFScore = zeros(u,1);
bestFScore = zeros(u,1);
bestFScore = bestFScore - inf;
paramNameOrder = '#neuronIndex, mu, gama';
bestParameter = cell(u,1);
clusteringResult = cell(u,diffParamNum,length(mu),length(gama),seedSetNum);
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
    i
    for seedSetCount=1:seedSetNum
        seedSetCount
        S = seedDataMat{i,seedSetCount};
        [~,k] = size(S);

        userTrueC = trueC(find(cluster_userIndex(:,1) == i-1),:)'; %transform to n*k

        % SEC
        for paramCount=1:diffParamNum
            for muCount=1:length(mu)
                for gamaCount=1:length(gama)
                    highestFScore = 0;
                    for initIdx=1:initIterNum
                        try
                        newF = SEC(real(userDataMat{i}{paramCount,2})',k, mu(muCount), gama(gamaCount)); % eat d*n, return n*k

                        %seedOKM
                        clustering = seedOKM(newF,k,S',0.0001); %return k*n
                        result = clustering'; %transform to n*k
                        
                        % calculate the recall and precision
                        base = sum(sum(userTrueC));
                        intersect = userTrueC & result;
                        theRecall = sum(sum(intersect))/base;
                        theFilterPrecision = filteredPrecision(userTrueC, result);
                        TheFScore = 2*((theRecall*theFilterPrecision)/(theRecall+theFilterPrecision))
                        
                        if(highestFScore < TheFScore)
                            highestFScore = TheFScore;
                            clusteringResult{i,paramCount,muCount,gamaCount,paramCount} = {result};
                            
                            recall(i,paramCount,muCount,gamaCount,seedSetCount) = theRecall;
                            filterPrecision(i,paramCount,muCount,gamaCount,seedSetCount) = theFilterPrecision;
                            FScore(i,paramCount,muCount,gamaCount,seedSetCount) = TheFScore;
                            
                        end
                        catch exception
                            continue;
                        end
                    end
                end
            end
        end
    end
    for paramCount=1:diffParamNum
        for muCount=1:length(mu)
            for gamaCount=1:length(gama)
                avgF = mean(FScore(i,paramCount,muCount,gamaCount,:));
                if(bestFScore(i,1) < avgF)
                    bestFScore(i,1) = avgF;
					recallsOfBestFScore(i,1) = mean(recall(i,paramCount,muCount,gamaCount,:));
					precisionsOfBestFScore(i,1) = mean(filterPrecision(i,paramCount,muCount,gamaCount,:));
                    bestParameter{i,1} = [paramCount,muCount,gamaCount];
                end
            end
        end
    end
	
    save([outputPath algoResultMat num2str(fromUser) '-' num2str(toUser) '_SEC_' fileParamName '_with_SeedOKM_recall_precision.mat'], 'recallsOfBestFScore','precisionsOfBestFScore', 'recall','filterPrecision','FScore','bestFScore','bestParameter','paramNameOrder');
end

avgR = mean(recallsOfBestFScore(find(recallsOfBestFScore>0)));
avgP = mean(precisionsOfBestFScore(find(precisionsOfBestFScore>0)));

disp(['recall: ' num2str(avgR)]);
disp(['precision: ' num2str(avgP)]);
disp(['FScore: ' num2str(2*((avgR*avgP)/(avgR+avgP)))]);
