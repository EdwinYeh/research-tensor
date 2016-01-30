sigmaList = [50, 100, 250, 500, 1000, 10000];

for sigmaTryTime = 1:6
    datasetId = 1;
    SetParameter;
    lambdaTryTime = 4;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_allfree_cvx_sigma_%f', datasetId, sigma);
    PrepareExperiment2;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    Main_ours_allfree_cvx;
end