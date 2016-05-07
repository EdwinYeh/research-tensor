function predictAndOutputResult(algorithmName,  isTestPhase)

for datasetId = 1:6
    fprintf('datasetId: %d\n', datasetId);
    if datasetId <= 6
        labelPrefixDirectory = '../20-newsgroup/';
    elseif datasetId > 6 && datasetId <=9
        labelPrefixDirectory = '../Reuter/';
    elseif datasetId >= 10 && datasetId <= 13
        labelPrefixDirectory = '../Animal_img/';
    end
    
    sampleSourceDataIndex = csvread(sprintf('sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
    sampleValidationDataIndex = csvread(sprintf('sampleIndex/sampleValidationDataIndex%d.csv', datasetId));
    sampleTestDataIndex = csvread(sprintf('sampleIndex/sampleTestDataIndex%d.csv', datasetId));
    numValidationInstance = length(sampleValidationDataIndex);
    numTestInstance = length(sampleTestDataIndex);
    if isTestPhase
        numTargetInstance = numValidationInstance + numTestInstance;
    else
        numTargetInstance = numValidationInstance;
    end
    domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};
    allLabel = cell(1,2);
    for dom = 1:2
        domainName = domainNameList{dom};
        allLabel{dom} = load([labelPrefixDirectory, domainName(1:length(domainName)-4), '_label.csv']);
    end

    sourceDomainLabel = allLabel{1}(sampleSourceDataIndex, :);
    targetDomainLabel = [allLabel{2}(sampleValidationDataIndex, :); allLabel{2}(sampleTestDataIndex, :)];
    
    UDirectoryPrefix = sprintf('../exp_result/%s/%d/', algorithmName, datasetId);
    disp(UDirectoryPrefix);
    FileNames = dir(UDirectoryPrefix);
    numFiles = length(FileNames) - 3;
    resultFileGaussin = fopen(sprintf('../exp_result/%s%d_gaussian.csv', algorithmName,   datasetId), 'w');
    fprintf(resultFileGaussin, 'accuracy,objective,cpRank,instanceCluster,featureCluster,sigma,sigma2lambda,gama,delta\n');

    for fileId = 1:numFiles
        fprintf('dataset %d, file %d\n', datasetId, fileId);
        fileName = FileNames(fileId + 3).name;
        fileParameters = strsplit(fileName, '_');
        tmpSplitArray = strsplit(fileParameters{8}, '.');
        fileParameters{8} = tmpSplitArray{1};
        UDirectory = sprintf('%s%s', UDirectoryPrefix, fileName);
        load(UDirectory);
        if isTestPhase
             CVFoldSize = (numValidationInstance+numTestInstance)/5;
             holdoutDataIndex = 1:CVFoldSize;
        else
            CVFoldSize = numValidationInstance/5;
            holdoutDataIndex = 1:CVFoldSize;
        end
        numCorrectPredict = 0;
        for fold = 1: 5
            targetTrainingDataIndex = setdiff(1:numTargetInstance,holdoutDataIndex);
            trainingData = [U{1}; U{2}(targetTrainingDataIndex,:)];
            trainingLabel = [sourceDomainLabel; targetDomainLabel(targetTrainingDataIndex, :)];
            svmModel = fitcsvm(trainingData, trainingLabel, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
            predictLabel = predict(svmModel, U{2}(holdoutDataIndex,:));
            for dataIndex = 1: 100
                if targetDomainLabel(holdoutDataIndex(dataIndex)) == predictLabel(dataIndex)
                    numCorrectPredict = numCorrectPredict + 1;
                end
            end
            holdoutDataIndex = holdoutDataIndex + CVFoldSize;
        end
        accuracy = numCorrectPredict/ (numTargetInstance);
        try
            fprintf(resultFileGaussin, '%g,%g,%s,%s,%s,%s,%s,%s,%s,%s\n', accuracy, newObjectiveScore, fileParameters{2}, fileParameters{3}, fileParameters{4}, fileParameters{5}, fileParameters{6}, fileParameters{7}, fileParameters{8});
        catch exception
            disp(exception.message);
        end
    end
    fclose(resultFileGaussian);
end
end