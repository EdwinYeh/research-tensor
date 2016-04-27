function predictAndOutputResult2(resultDirectoryName)

for datasetId = 1:13
    numSampleData = 500;
    fprintf('datasetId: %d\n', datasetId);
    if datasetId <= 6
        labelPrefixDirectory = '../20-newsgroup/';
    elseif datasetId > 6 && datasetId <=9
        labelPrefixDirectory = '../Reuter/';
    elseif datasetId >= 10 && datasetId <= 13
        labelPrefixDirectory = '../Animal_img/';
    end
    
    sampleSourceDataIndex = csvread(sprintf('sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
    sampleTargetDataIndex = csvread(sprintf('sampleIndex/sampleTargetDataIndex%d.csv', datasetId));
    domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};
    allLabel = cell(1,2);
    for dom = 1:2
        domainName = domainNameList{dom};
        allLabel{dom} = load([labelPrefixDirectory, domainName(1:length(domainName)-4), '_label.csv']);
    end
    sourceDomainLabel = allLabel{1}(sampleSourceDataIndex, :);
    targetDomainLabel = allLabel{2}(sampleTargetDataIndex, :);
    
    UDirectoryPrefix = sprintf('../exp_result/%s/%d/', resultDirectoryName, datasetId);
    
    FileNames = dir(UDirectoryPrefix);
    numFiles = length(FileNames) - 3;
    resultFile = fopen(sprintf('../exp_result/%s%d.csv', resultDirectoryName,   datasetId), 'w');
    fprintf(resultFile, 'accuracy, lambda, gama, sigma, instanceCluster, featureCluster, cpRank\n');
    for fileId = 1:numFiles
        fprintf('dataset %d, file %d\n', datasetId, fileId);
        fileName = FileNames(fileId + 3).name;
        fileParameters = strsplit(fileName, '_');
        UDirectory = sprintf('%s%s', UDirectoryPrefix, fileName);
        U = load(UDirectory);
        targetTestingDataIndex = 1:100;
        numCorrectPredict = 0;
        for fold = 1: 5
            targetTrainingDataIndex = setdiff(1:500,targetTestingDataIndex);
            trainingData = [U{1}; U{2}(targetTrainingDataIndex,:)];
            trainingLabel = [sourceDomainLabel; targetDomainLabel(targetTrainingDataIndex, :)];
            svmModel = fitcsvm(trainingData, trainingLabel, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
            predictLabel = predict(svmModel, U{2}(targetTestingDataIndex,:));
            for dataIndex = 1: 100
                if targetDomainLabel(targetTestingDataIndex(dataIndex)) == predictLabel(dataIndex)
                    numCorrectPredict = numCorrectPredict + 1;
                end
            end
            targetTestingDataIndex = targetTestingDataIndex + 100;
        end
        accuracy = numCorrectPredict/ (numSampleData);
        try
            fprintf(resultFile, '%f,%s,%s,%s,%s,%s,%s\n', accuracy, fileParameters{2}, fileParameters{3}, fileParameters{4}, fileParameters{5}, fileParameters{6}, fileParameters{7});
        catch exception
            disp(exception.message);
        end
    end
    fclose(resultFile);
end
end