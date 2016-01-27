sigmaList = [1, 10, 100, 1000, 10000];

for sigmaTryTime = 1:5
%     datasetId = 1;
    SetParameter;
    lambdaTryTime = 6;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_sigma_normc_%f', datasetId, sigma);
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    PrepareExperiment_normc;
    main_ours;
end