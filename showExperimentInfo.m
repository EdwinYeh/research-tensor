function showExperimentInfo(experimentTitle, datasetId, prefix, numSourceList, numTargetList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature, featureCluster)
    fprintf('Title: %s\n', experimentTitle);
    fprintf('Dataset: %s\n', prefix);
    fprintf('Source domain: source%d.csv\n', datasetId);
    fprintf('Target domain: target%d.csv\n', datasetId);
    fprintf('Number of instances: [%d %d]\n', numSourceList(datasetId), numTargetList(datasetId));
    fprintf('Number of features: [%d %d]\n', numSourceFeatureList(datasetId), numTargetFeatureList(datasetId));
    fprintf('Number of sample instances: [%d %d]\n', numSampleInstance(1), numSampleInstance(2));
    fprintf('Number of sample features: [%d %d]\n', numSampleFeature(1), numSampleFeature(2));
    fprintf('featureCluster: %d\n', featureCluster);
end