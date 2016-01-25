sigmaList = [0.2, 0.4, 0.6, 0.8, 1];

for sigmaTryTime = 1:5
%     datasetId = 1;
    SetParameter;
    lambdaTryTime = 0;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_sigma_normc_%f', datasetId, sigma);
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    PrepareExperiment_normc;
    main_ours;
end