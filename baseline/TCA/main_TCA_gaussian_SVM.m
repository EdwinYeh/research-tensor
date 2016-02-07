% datasetId = 6;
numDom = 2;
mu = 1;
numSampleFeature = 2000;
numSourceData = 500;
numValidateData = 100;
numTestData = 500;
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
elseif datasetId >= 10
    dataType = 2;
    prefix = '../../../Animal_img/';
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
% numSourceData = sizeOfSourceDomainData(1);
% numTargetData = sizeOfTargetDomainData(1);
% 
% sampleTargetAndTestDataIndex = randperm(numTargetData, (numSampleData+numTestData));
% sampleSourceDataIndex = randperm(numSourceData, numSampleData);
% sampleTargetDataIndex = sampleTargetAndTestDataIndex(1:numSampleData);
% sampleTestDataIndex = sampleTargetAndTestDataIndex((numSampleData+1):(numSampleData+numTestData));
sampleSourceDataIndex = csvread(sprintf('../../sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
sampleValidateDataIndex = csvread(sprintf('../../sampleIndex/sampleValidateDataIndex%d.csv', datasetId));
sampleTestDataIndex = csvread(sprintf('../../sampleIndex/sampleTargetDataIndex%d.csv', datasetId));

testData = targetDomainData(sampleTestDataIndex, :);
sourceDomainData = sourceDomainData(sampleSourceDataIndex, :);
targetDomainData = targetDomainData(sampleValidateDataIndex, :);

testData = normr(testData);
sourceDomainData = normr(sourceDomainData);
targetDomainData = normr(targetDomainData);

testY = targetY(sampleTestDataIndex);
sourceY = sourceY(sampleSourceDataIndex);
targetY = targetY(sampleValidateDataIndex);
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
    [~, avgEmpError, validationAccuracy] = trainAndCvGaussianTCA(mu, sigma, numFold, numSourceData, numValidateData, featureDimAfterReduce, sourceDomainData, targetDomainData, Y);
    if validationAccuracy > bestValidationAccuracy
        bestValidationAccuracy = validationAccuracy;
        bestMu = mu;
        bestSigma = sigma;     
    end
    fprintf(resultFile, '%f,%f,%f,%f\n', mu, sigma, avgEmpError, validationAccuracy);
    end
end
Y = [sourceY; testY];
[predictLabel, avgEmpError, accuracy] = trainAndCvGaussianTCA(bestMu, bestSigma, numFold, numSourceData, numTestData, featureDimAfterReduce, sourceDomainData, testData, Y); 
csvwrite(sprintf('../../../exp_result/predict_result/TCA_gaussian%d_predict_result.csv', datasetId), predictLabel);
fprintf(resultFile, '%f,%f,%f,%f\n', bestMu, bestSigma, avgEmpError, accuracy);
fclose(resultFile);