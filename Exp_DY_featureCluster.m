disp('DY_featureCluster');
datasetId = 1;
SetParameter;
sampleSizeLevel = '';
resultDirectory = sprintf('../exp_result/featureCluster/DY/%d/', datasetId);
mkdir(resultDirectory);
expTitle = sprintf('DY_%d', datasetId);
sigma = 0.015;
lambda = 0.0001;
delta = 10^-13;
cpRank = 50;
numInstanceCluster = 10;
numFeatureClusterList = [5, 10, 15, 30, 50, 100];
isTestPhase = true;
randomTryTime = 5;
nuCVFold = 1;
resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'w');
fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,lambda,delta,objectiveScore,accuracy,trainingTime\n');
PrepareExperiment;
for tuneFeatureCluster = 1: length(numFeatureClusterList)
    numFeatureCluster = numFeatureClusterList(tuneFeatureCluster);
    main_DY;
end
fclose(resultFile);
