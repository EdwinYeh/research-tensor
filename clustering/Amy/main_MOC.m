
algoResultMat = '';
load([outputPath algoResultMat '.mat']);
load([parentPath inputPath datasetName '.mat']);
trueC = cluster_data;

u = length(userDataMat);
fromUser = 1;
toUser = u;


[diffParamNum,~] = size(userDataMat{1,1});
[~,seedSetNum] = size(seedDataMat);
recall = zeros(u,diffParamNum,seedSetNum);
filterPrecision = zeros(u,diffParamNum,seedSetNum);
FScore = zeros(u,diffParamNum,seedSetNum);
recallsOfBestFScore = zeros(u,1);
precisionsOfBestFScore = zeros(u,1);
bestFScore = zeros(u,1);
bestFScore = bestFScore - inf;
bestParameter = ones(u,1);

clusteringResult = cell(u,diffParamNum,seedSetNum);
initIterNum = 2;

for i=fromUser:toUser
    i
    for seedSetCount=1:seedSetNum
        S = seedDataMat{i,seedSetCount};
        [~,k] = size(S);

        userTrueC = trueC(find(cluster_userIndex(:,1) == i-1),:)'; %transform to n*k

        % SeedMOC
        for paramCount=1:diffParamNum
            highestFScore = 0;
            for initIdx=1:initIterNum
                try
                clustering = seedMOC(userDataMat{i}{paramCount,2}, k, S'); %return k*n
                result = clustering'; %transform to n*k
                
                % calculate the recall and precision
                base = sum(sum(userTrueC));
                intersect = userTrueC & result;
                theRecall = sum(sum(intersect))/base;
                theFilterPrecision = filteredPrecision(userTrueC, result);
                TheFScore = 2*((theRecall*theFilterPrecision)/(theRecall+theFilterPrecision))
                
                if(highestFScore < TheFScore)
                    highestFScore = TheFScore;
                    clusteringResult{i,paramCount,seedSetCount} = result;
                    
                    recall(i,paramCount,seedSetCount) = theRecall;
                    filterPrecision(i,paramCount,seedSetCount) = theFilterPrecision;
                    FScore(i,paramCount,seedSetCount) = TheFScore;
                    
                end

                catch exception
                    continue;
                end
            end
        end
    end
    for paramCount=1:diffParamNum
        avgF = mean(FScore(i,paramCount,:));
        if(bestFScore(i,1) < avgF)
            bestFScore(i,1) = avgF;
			recallsOfBestFScore(i,1) = mean(recall(i,paramCount,:));
			precisionsOfBestFScore(i,1) = mean(filterPrecision(i,paramCount,:));
            bestParameter(i,1) = paramCount;
        end
    end
	
    save([outputPath algoResultMat num2str(fromUser) '-' num2str(toUser) '_MOC_recall_precision.mat'],'recallsOfBestFScore','precisionsOfBestFScore', 'recall','filterPrecision','FScore','bestFScore','bestParameter','clusteringResult');
end

avgR = mean(recallsOfBestFScore(find(recallsOfBestFScore>0)));
avgP = mean(precisionsOfBestFScore(find(precisionsOfBestFScore>0)));

disp(['recall: ' num2str(avgR)]);
disp(['precision: ' num2str(avgP)]);
disp(['FScore: ' num2str(2*((avgR*avgP)/(avgR+avgP)))]);
