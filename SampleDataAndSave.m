function sampleDataAndSave(datasetId, numValidationInstance, numTestInstance)

numDom = 2;
sourceDomain = 1;
targetDomain = 2;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
elseif datasetId > 6 && datasetId <= 9
    dataType = 1;
    prefix = '../Reuter/';
elseif datasetId >= 10 &&datasetId <=13
    dataType = 2;
    prefix = '../Animal_img/';
elseif datasetId >=14 && datasetId <=23
    prefix = '../DBLP/';
end

if datasetId <= 13
    domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};
    X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);
    [numSourceInstance, ~] = size(X{sourceDomain});
    [numTargetInstance, ~] = size(X{targetDomain});
    fprintf('dataset:%d=>[%d,%d]\n', datasetId, numSourceInstance, numTargetInstance);
    
elseif datasetId >=14 && datasetId <=23
    fileName = sprintf('%sDBLP%d.mat', prefix, datasetId - 13);
    load(fileName);
    [numSourceInstance, ~] = size(edges1);
    [numTargetInstance, ~] = size(edges2);
    
end

sampleSourceDataIndex = randperm(numSourceInstance, numValidationInstance(sourceDomain));
sampleTargetDataIndex = randperm(numTargetInstance, numValidationInstance(targetDomain) + numTestInstance);
sampleTestDataIndex = sampleTargetDataIndex(numValidationInstance(targetDomain)+1:numValidationInstance(targetDomain)+numTestInstance);
sampleValidationDataIndex = sampleTargetDataIndex(1:numValidationInstance(targetDomain));

csvwrite(sprintf('sampleIndex/sampleSourceDataIndex%d.csv', datasetId), sampleSourceDataIndex);
csvwrite(sprintf('sampleIndex/sampleValidationDataIndex%d.csv', datasetId), sampleValidationDataIndex);
csvwrite(sprintf('sampleIndex/sampleTestDataIndex%d.csv', datasetId), sampleTestDataIndex);

end