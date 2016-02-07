function sampleDataAndSave(datasetId)

numSampleInstance = [500, 500];
numDom = 2;
sourceDomain = 1;
targetDomain = 2;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
elseif datasetId > 6 && datasetId <= 9
    dataType = 1;
    prefix = '../Reuter/';
elseif datasetId >= 10
    dataType = 2;
    prefix = '../Animal_img/';
end

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};
X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);

[numSourceInstance, ~] = size(X{sourceDomain});
[numTargetInstance, ~] = size(X{targetDomain});
sampleSourceDataIndex = randperm(numSourceInstance, numSampleInstance(sourceDomain));
sampleTargetDataIndex = randperm(numTargetInstance, numSampleInstance(targetDomain)+100);
sampleValidateDataIndex = sampleTargetDataIndex(501:600);
sampleTargetDataIndex = sampleTargetDataIndex(1:500);
csvwrite(sprintf('sampleIndex/sampleSourceDataIndex%d.csv', datasetId), sampleSourceDataIndex);
csvwrite(sprintf('sampleIndex/sampleTargetDataIndex%d.csv', datasetId), sampleTargetDataIndex);
csvwrite(sprintf('sampleIndex/sampleValidateDataIndex%d.csv', datasetId), sampleValidateDataIndex);

end