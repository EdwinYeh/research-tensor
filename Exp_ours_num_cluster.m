
for sigmaTryTime = 1:5
    datasetId = 1;
    SetParameter;
    sigma = 0.5;
    lambdaTryTime = 0;
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    exp_title = sprintf('ours_%d_num_cluster', datasetId, sigma);
    showExperimentInfo(datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    PrepareExperiment;
    main_ours;
end