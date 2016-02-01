sigmaList = [100, 150, 200, 250, 300, 350, 400, 450, 500];

for sigmaTryTime = 1:8
    datasetId = 1;
    SetParameter;
    lambdaTryTime = 3;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_allfree_cvx_sigma_%f', datasetId, sigma);
    PrepareExperiment2;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    Main_ours_allfree_cvx;
end