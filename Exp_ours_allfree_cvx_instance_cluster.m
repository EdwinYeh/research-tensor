fprintf('Start testing\n');
exp_title = sprintf('ours_%d_instance_cluster', datasetId);
SetParameter;
isTestPhase = true;
randomTryTime = 1;
sigma = 0.1;
lambda = 0.000001;

instanceClusterList = [2, 4, 8, 16, 32];
for instanceClusterTryTime = 1:5
    numInstanceCluster = instanceClusterList(instanceClusterTryTime);
    PrepareExperiment2;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    main_ours_allfree_cvx;
end