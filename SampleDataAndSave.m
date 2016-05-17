function sampleDataAndSave(datasetId, saveDirectory, numInstance, numTestInstance)

mkdir(sprintf('sampleIndex/%s', saveDirectory));

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

sampleSourceDataIndex = randperm(numSourceInstance, numInstance(sourceDomain));
sampleTargetDataIndex = randperm(numTargetInstance, numInstance(targetDomain));
sampleTestDataIndex = sampleTargetDataIndex(numInstance(targetDomain)-numTestInstance+1:numInstance(targetDomain));
sampleValidationDataIndex = sampleTargetDataIndex(1:numInstance(targetDomain)-numTestInstance);

csvwrite(sprintf('sampleIndex/%ssampleSourceDataIndex%d.csv', saveDirectory, datasetId), sampleSourceDataIndex);
csvwrite(sprintf('sampleIndex/%ssampleValidationDataIndex%d.csv', saveDirectory, datasetId), sampleValidationDataIndex);
csvwrite(sprintf('sampleIndex/%ssampleTestDataIndex%d.csv', saveDirectory, datasetId), sampleTestDataIndex);

end