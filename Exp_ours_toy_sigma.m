sigmaList = [0.1, 1, 10, 50, 100, 250];

for sigmaTryTime = 1:6
    SetParameter;
    numSampleFeature = 2;
    lambdaTryTime = 3;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_sigma_%f', datasetId, sigma);
    PrepareExperiment;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    Main_ours;
end