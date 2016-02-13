function sampleDataAndSave(datasetId)

numSampleInstance = [500, 500];
numValidateInstance = 100;
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
    
elseif datasetId >=14 && datasetId <=23
    fileName = sprintf('%sDBLP%d.mat', prefix, datasetId - 13);
    load(fileName);
    [numSourceInstance, ~] = size(edges1);
    [numTargetInstance, ~] = size(edges2);
    
end

sampleSourceDataIndex = randperm(numSourceInstance, numSampleInstance(sourceDomain));
sampleTargetDataIndex = randperm(numTargetInstance, numSampleInstance(targetDomain)+100);
sampleValidateDataIndex = sampleTargetDataIndex(numSampleInstance(targetDomain)+1:numSampleInstance(targetDomain)+numValidateInstance);
sampleTargetDataIndex = sampleTargetDataIndex(1:numSampleInstance(targetDomain));
csvwrite(sprintf('sampleIndex/sampleSourceDataIndex%d.csv', datasetId), sampleSourceDataIndex);
csvwrite(sprintf('sampleIndex/sampleTargetDataIndex%d.csv', datasetId), sampleTargetDataIndex);
csvwrite(sprintf('sampleIndex/sampleValidateDataIndex%d.csv', datasetId), sampleValidateDataIndex);

end