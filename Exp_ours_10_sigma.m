clear;
clc;
sigmaList = [1, 5, 10, 20];

for sigmaTryTime = 1:4
    datasetId = 10;
    SetParameter;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('Motar2_W_%d_sigma_%f', datasetId, sigma);
    showExperimentInfo(datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    PrepareExperiment;
    main_ours;
end