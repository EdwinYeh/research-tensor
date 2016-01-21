clear;
clc;
sigmaList = [0.5, 1, 5, 10, 20];

for sigmaTryTime = 2:5
    datasetId = 1;
    SetParameter;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_sigma_%f_no_regular', datasetId, sigma);
    showExperimentInfo(datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    PrepareExperiment;
    main_ours_no_regular;
end