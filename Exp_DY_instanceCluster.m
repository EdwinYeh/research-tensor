disp('DY_instanceCluster');
datasetId = 1;
SetParameter;
sampleSizeLevel = '';
resultDirectory = sprintf('../exp_result/instanceCluster/DY/%d/', datasetId);
mkdir(resultDirectory);
expTitle = sprintf('DY_%d', datasetId);
sigma = 0.015;
lambda = 0.0001;
delta = 10^-13;
cpRank = 50;
numInstanceClusterList = [5, 10, 15, 30, 50, 100];
numFeatureCluster = 10;
isTestPhase = true;
randomTryTime = 5;
nuCVFold = 1;
resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'w');
fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,lambda,delta,objectiveScore,accuracy,trainingTime\n');
PrepareExperiment;
for tuneInstanceCluster = 1: length(numInstanceClusterList)
    numInstanceCluster = numInstanceClusterList(tuneInstanceCluster);
    main_DY;
end
fclose(resultFile);
