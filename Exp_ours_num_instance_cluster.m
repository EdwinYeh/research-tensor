numInstanceClusterList = [2, 4, 8, 16, 32];

for instanceClusterTryTime = 1:5
%     datasetId = 1;
    SetParameter;
    sigma = 0.5;
    lambdaTryTime = 0;
    numInstanceCluster = numInstanceClusterList(instanceClusterTryTime);
    numFeatureCluster = 5;
    exp_title = sprintf('ours_%d_num_instance_cluster_%d', datasetId, numInstanceCluster);
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    PrepareExperiment;
    main_ours;
end