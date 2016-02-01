sigmaList = [0.1, 1, 10, 50, 100, 250];

for sigmaTryTime = 1:6
%     datasetId = 1;
    SetParameter;
    lambdaTryTime = 3;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_allfree_cvx_sigma_%f', datasetId, sigma);
    PrepareExperiment2;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    main_ours_allfree_cvx;
end