fprintf('Start testing\n');
exp_title = sprintf('ours_%d_instance_cluster', datasetId);
SetParameter;
isTestPhase = true;
randomTryTime = 1;
sigma = 0.1;
lambda = 0.000001;

featureClusterList = [2, 4, 8, 16, 32];
for featureClusterTryTime = 1:5
    numFeatureCluster = featureClusterList(featureClusterTryTime);
    PrepareExperiment2;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    main_ours_allfree_cvx;
end