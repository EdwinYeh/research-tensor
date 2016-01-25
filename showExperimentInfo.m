function showExperimentInfo(exp_title, datasetId, prefix, numInstance, numFeature, numInstanceCluster, numFeatureCluster, sigma)
    time = round(clock);
    fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
    fprintf('Title: %s\n', exp_title);
    fprintf('Dataset: %s\n', prefix);
    fprintf('Source domain: source%d.csv\n', datasetId);
    fprintf('Target domain: target%d.csv\n', datasetId);
    fprintf('Number of instances: [%d %d]\n', numInstance(1), numInstance(2));
    fprintf('Number of features: %d\n', numFeature);
    fprintf('Number of instance cluster: %d\n', numInstanceCluster);
    fprintf('Number of feature cluster: %d\n', numFeatureCluster);
    fprintf('Sigma of gaussian: %f\n', sigma);
end