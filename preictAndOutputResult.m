function preictAndOutputResult(resultDirectoryName)

    for datasetId = 1:13
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
        
        lambdaTryTime = 3;
        gamaTryTime = 3;
        for tuneLambda= 0: lambdaTryTime
            lambda = 0.000001* (100^ tuneLambda);
            for tuneGama= 0: gamaTryTime
                gama = 0.000001* (100^ tuneGama);
                fprintf('(lambda, gama)=(%f,%f)\n', lambda, gama);
                UDirectory = sprintf('%sU_%f_%f.mat', UDirectoryPrefix, lambda, gama);
                load(UDirectory);
                targetTestingDataIndex = 1:100;
                numCorrectPredict = 0;
                for fold = 1: 5
                    targetTrainingDataIndex = setdiff(1:500,targetTestingDataIndex);
                    trainingData = [bestU{1}; bestU{2}(targetTrainingDataIndex,:)];
                    trainingLabel = [sourceDomainLabel; targetDomainLabel(targetTrainingDataIndex, :)];
                    svmModel = fitcsvm(trainingData, trainingLabel, 'KernelFunction', 'rbf');
                    predictLabel = predict(svmModel, bestU{2}(targetTestingDataIndex,:));
                    for dataIndex = 1: 100
                        if targetDomainLabel(targetTestingDataIndex(dataIndex)) == predictLabel(dataIndex)
                            numCorrectPredict = numCorrectPredict + 1;
                        end
                    end
                    targetTestingDataIndex = targetTestingDataIndex + 100;
                end
                accuracy = numCorrectPredict/ (500);
                resultFile = fopen(sprintf('../exp_result/DX%d.csv', datasetId), 'a');
                fprintf(resultFile, '%f,%f,%f\n', lambda, gama, accuracy);
                fclose(resultFile);
            end
        end
    end
end