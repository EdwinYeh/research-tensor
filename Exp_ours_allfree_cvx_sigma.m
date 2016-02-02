sigmaList = [0.1, 0.2, 0.3, 0.4, 0.5];

for sigmaTryTime = 1:5
%     datasetId = 1;
    SetParameter;
    lambdaTryTime = 6;
    sigma = sigmaList(sigmaTryTime);
    exp_title = sprintf('ours_%d_allfree_cvx_sigma_%f', datasetId, sigma);
    PrepareExperiment2;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    main_ours_allfree_cvx;
end