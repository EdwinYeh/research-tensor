sigmaList = [1, 10, 100, 1000, 10000];

for sigmaTryTime = 1:5
%     datasetId = 1;
    SetParameter;
    lambdaTryTime = 6;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_sigma_%f', datasetId, sigma);
    PrepareExperiment;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    Main_ours;
end