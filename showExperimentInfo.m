function showExperimentInfo(experimentTitle, datasetId, prefix, numInstance, numFeature)
    time = round(clock);
    fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
    fprintf('Title: %s\n', experimentTitle);
    fprintf('Dataset: %s\n', prefix);
    fprintf('Source domain: source%d.csv\n', datasetId);
    fprintf('Target domain: target%d.csv\n', datasetId);
    fprintf('Number of instances: [%d %d]\n', numInstance(1), numInstance(2));
    fprintf('Number of features: [%d %d]\n', numFeature(1), numFeature(2));
end