clear;
clc;
% sigmaList = [2.5, 5, 7.5, 10];

for sigmaTryTime = 1:1
    SetParameter;
    lambdaTryTime = 0;
%     sigma = sigmaList(sigmaTryTime);
    sigma = 5;
    exp_title = sprintf('ours_%d_allfree', datasetId);
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    PrepareExperiment2;
    Main_ours_allfree;
end