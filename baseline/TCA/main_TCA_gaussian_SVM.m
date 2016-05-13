% % % datasetId = 6;
numDom = 2;
numSampleFeature = 3000;
% numSourceData = 500;
% numValidateData = 100;
% numTestData = 500;
numFold = 5;
featureDimAfterReduce = 15;
randomTryTime = 5;

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
% numSourceData = sizeOfSourceDomainData(1);
% numTargetData = sizeOfTargetDomainData(1);
%
% sampleTargetAndTestDataIndex = randperm(numTargetData, (numSampleData+numTestData));
% sampleSourceDataIndex = randperm(numSourceData, numSampleData);
% sampleTargetDataIndex = sampleTargetAndTestDataIndex(1:numSampleData);
% sampleTestDataIndex = sampleTargetAndTestDataIndex((numSampleData+1):(numSampleData+numTestData));
sampleSourceDataIndex = csvread(sprintf('../../sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
sampleValidationDataIndex = csvread(sprintf('../../sampleIndex/sampleValidationDataIndex%d.csv', datasetId));
sampleTestDataIndex = csvread(sprintf('../../sampleIndex/sampleTestDataIndex%d.csv', datasetId));

numSourceData = length(sampleSourceDataIndex);
numValidateData = length(sampleValidationDataIndex);
numTestData = length(sampleTestDataIndex);

testData = targetDomainData(sampleTestDataIndex, :);
sourceDomainData = sourceDomainData(sampleSourceDataIndex, :);
targetDomainData = targetDomainData(sampleValidationDataIndex, :);

testData = normr(testData);
sourceDomainData = normr(sourceDomainData);
targetDomainData = normr(targetDomainData);

testY = targetY(sampleTestDataIndex);
sourceY = sourceY(sampleSourceDataIndex);
targetY = targetY(sampleValidationDataIndex);
Y = [sourceY; targetY];

resultDirectory = sprintf('../../../exp_result/TCA/%d/', datasetId);
mkdir(resultDirectory);
% resultFile = fopen(sprintf('%sresult_TCA_validation%d.csv', resultDirectory, datasetId), 'w');
% fprintf(resultFile, 'mu,sigma,accuracy\n');
% 
% bestValidationAccuracy = 0;
% bestMu = -1;
% bestSigma = -1;
% for tuneMu = 0:4
%     for tuneSigma = 0:3
%         mu = 0.0001 * 100 ^ tuneMu;
%         sigma = 0.001 * 10 ^ tuneSigma;
%         try
%             [~, ~, validationAccuracy] = trainAndCvGaussianTCA(mu, sigma, numFold, numSourceData, numValidateData, -1, featureDimAfterReduce, sourceDomainData, targetDomainData, Y, false);
%             if validationAccuracy > bestValidationAccuracy
%                 bestValidationAccuracy = validationAccuracy;
%                 bestMu = mu;
%                 bestSigma = sigma;
%             end
%             fprintf(resultFile, '%f,%f,%f\n', mu, sigma, validationAccuracy);
%         catch exception
%             disp(exception.message);
%             continue;
%         end
%     end
% end
% fclose(resultFile);
bestSigma = 0.1;
bestMu = 1;
resultFile = fopen(sprintf('%sresult_TCA_test%d.csv', resultDirectory, datasetId), 'w');
fprintf(resultFile, 'mu,sigma,accuracy,time\n');
Y = [sourceY; [targetY; testY]];
targetDomainData = [targetDomainData; testData];
for t = 1:randomTryTime
    totalTimer = tic;
    [predictLabel, avgEmpError, accuracy] = trainAndCvGaussianTCA(bestMu, bestSigma, 1, numSourceData, numValidateData, numTestData, featureDimAfterReduce, sourceDomainData, targetDomainData, Y, true);
    totalTime = toc(totalTimer);
    % csvwrite(sprintf('../../../exp_result/predict_result/TCA_gaussian%d_predict_result.csv', datasetId), predictLabel);
    fprintf(resultFile, '%f,%f,%f,%f\n', bestMu, bestSigma, accuracy, totalTime);
end
fclose(resultFile);