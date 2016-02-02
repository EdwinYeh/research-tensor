% datasetId = 6;
numDom = 2;
mu = 1;
numSampleFeature = 2;
numSampleData = 500;
numTestData = 250;
numFold = 5;
featureDimAfterReduce = 2;

fprintf('datasetId: %d\n', datasetId);

% dataType:
% 1 means matrix form storage
% 2 means "x y value" form storage
if datasetId <= 6
    dataType = 1;
    prefix = '../../../20-newsgroup/';
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../../../Reuter/';
elseif datasetId == 10
    dataType = 2;
    prefix = '../../../Animal_img/';
elseif datasetId == 11 || datasetId == 12
    dataType = 2;
    prefix = '../../../Toy_dataset/'
end

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

% Load data from source and target domain data
X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);
for i = 1:numDom
    denseFeatureIndex = findDenseFeature(X{i}, numSampleFeature);
    X{i} = X{i}(:, denseFeatureIndex);
end
sourceY = load([prefix sprintf('source%d_label.csv', datasetId)]);
targetY = load([prefix sprintf('target%d_label.csv', datasetId)]);

sourceDomainData = X{1};
targetDomainData = X{2};
sizeOfSourceDomainData = size(X{1});
sizeOfTargetDomainData = size(X{2});
numSourceData = sizeOfSourceDomainData(1);
numTargetData = sizeOfTargetDomainData(1);

sampleTargetAndTestDataIndex = randperm(numTargetData, (numSampleData+numTestData));
sampleSourceDataIndex = randperm(numSourceData, numSampleData);
sampleTargetDataIndex = sampleTargetAndTestDataIndex(1:numSampleData);
sampleTestDataIndex = sampleTargetAndTestDataIndex((numSampleData+1):(numSampleData+numTestData));
% sampleSourceDataIndex = csvread(sprintf('%ssampleSourceIndex%d.csv', prefix, datasetId));
% sampleTargetDataIndex = csvread(sprintf('%ssampleTargetIndex%d.csv', prefix, datasetId));
% sampleTestDataIndex = csvread(sprintf('%ssampleTestIndex%d.csv', prefix, datasetId));
testData = targetDomainData(sampleTestDataIndex, :);
sourceDomainData = sourceDomainData(sampleSourceDataIndex, :);
targetDomainData = targetDomainData(sampleTargetDataIndex, :);

testData = normr(testData);
sourceDomainData = normr(sourceDomainData);
targetDomainData = normr(targetDomainData);

numSourceData = numSampleData;
numTargetData = numSampleData;

testY = targetY(sampleTestDataIndex);
sourceY = sourceY(sampleSourceDataIndex);
targetY = targetY(sampleTargetDataIndex);
Y = [sourceY; targetY];

resultFile = fopen(sprintf('result_TCA_gaussian%d.csv', datasetId), 'w');
fprintf(resultFile, 'mu,sigma,empError,accuracy\n');

bestValidationAccuracy = 0;
bestMu = 1;
bestSigma = 1;
for tuneMu = 0:3
    for tuneSigma = 0:3
    mu = 0.001 * 100 ^ tuneMu;
    sigma = 0.001 * 100 ^ tuneSigma;
    [avgEmpError, validationAccuracy] = trainAndCvGaussianTCA(mu, sigma, numFold, numSourceData, numTargetData, featureDimAfterReduce, sourceDomainData, targetDomainData, Y);
    if validationAccuracy > bestValidationAccuracy
        bestValidationAccuracy = validationAccuracy;
        bestMu = mu;
        bestSigma = sigma;
    end
    fprintf(resultFile, '%f,%f,%f,%f\n', mu, sigma, avgEmpError, validationAccuracy);
    end
end
Y = [sourceY; testY];
[avgEmpError, accuracy] = trainAndCvGaussianTCA(bestMu, bestSigma, numFold, numSourceData, numTestData, featureDimAfterReduce, sourceDomainData, testData, Y); 
fprintf(resultFile, '%f,%f,%f,%f\n', bestMu, bestSigma, avgEmpError, accuracy);
fclose(resultFile);