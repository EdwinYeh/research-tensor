function preictAndOutputResult(resultDirectoryName)

for datasetId = 7:13
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
    
    for fileId = 1:numFiles
        fprintf('dataset %d, file %d\n', datasetId, fileId);
        fileName = FileNames(fileId + 3).name;
        fileParameters = strsplit(fileName, '_');
        UDirectory = sprintf('%s%s', UDirectoryPrefix, fileName);
        load(UDirectory);
        targetTestingDataIndex = 1:100;
        numCorrectPredict = 0;
        for fold = 1: 5
            targetTrainingDataIndex = setdiff(1:500,targetTestingDataIndex);
            trainingData = [bestU{1}; bestU{2}(targetTrainingDataIndex,:)];
            trainingLabel = [sourceDomainLabel; targetDomainLabel(targetTrainingDataIndex, :)];
            svmModel = fitcsvm(trainingData, trainingLabel, 'KernelFunction', 'rbf','KernelScale','auto', 'Standardize', true);
            predictLabel = predict(svmModel, bestU{2}(targetTestingDataIndex,:));
            for dataIndex = 1: 100
                if targetDomainLabel(targetTestingDataIndex(dataIndex)) == predictLabel(dataIndex)
                    numCorrectPredict = numCorrectPredict + 1;
                end
            end
            targetTestingDataIndex = targetTestingDataIndex + 100;
        end
        accuracy = numCorrectPredict/ (500);
        try
            resultFile = fopen(sprintf('../exp_result/DX%d.csv', datasetId), 'a');
            fprintf(resultFile, '%f,%s,%s,%s,%s,%s,%s\n', accuracy, fileParameters{2}, fileParameters{3}, fileParameters{4}, fileParameters{5}, fileParameters{6}, fileParameters{7});
            fclose(resultFile);
        catch exception
            disp(exception.message);
            fclose(resultFile);
        end
    end
    
end
end