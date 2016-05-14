disp('cross_3ways');
SetParameter;
sampleSizeLevel = 'bigSample/';
resultDirectory = sprintf('../exp_result/timeissue/%d/', datasetId);
mkdir(resultDirectory);
expTitle = sprintf('DY_cross_3ways%d', datasetId);
sigma = 0.015;
lambdaList = 0.0001:0.0001:0.0003;
delta = 10^-13;
cpRank = 10;
numInstanceCluster = 10;
numFeatureCluster = 10;
isTestPhase = true;
randomTryTime = 1;
numCVFold = 1;
resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'w');
fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,lambda,delta,objectiveScore,accuracy,trainingTime\n');
PrepareExperiment;
for tuneLambda = 1:length(lambdaList)
    lambda = lambdaList(tuneLambda);
    main_DY_cross_domain_3way;
end
fclose(resultFile);