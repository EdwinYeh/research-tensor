disp('DY_cpRank');
datasetId = 1;
SetParameter;
sampleSizeLevel = '';
resultDirectory = sprintf('../exp_result/cpRank/DY/%d/', datasetId);
mkdir(resultDirectory);
expTitle = sprintf('DY_%d', datasetId);
sigma = 0.015;
lambda = 0.0001;
delta = 10^-13;
cpRankList = [5, 10, 15, 30, 50, 100];
numInstanceCluster = 15;
numFeatureCluster = 15;
isTestPhase = true;
randomTryTime = 5;
nuCVFold = 1;
resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'w');
fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,lambda,delta,objectiveScore,accuracy,trainingTime\n');
PrepareExperiment;
for tunecpRank = 1: length(cpRankList)
    cpRank = cpRankList(tunecpRank);
    main_DY;
end
fclose(resultFile);