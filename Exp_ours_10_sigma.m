clear;
clc;
sigmaList = [2.5, 5, 7.5, 10];

for sigmaTryTime = 1:4
    datasetId = 10;
    SetParameter;
    lambdaTryTime = 0;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_sigma_%f', datasetId, sigma);
    showExperimentInfo(datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    PrepareExperiment;
    main_ours;
end