datasetId = 10;
numDom = 2;
mu = 1;
numSampleFeature = 2000;
numSampleData = 500;
numFold = 5;
featureDimAfterReduce = 50;

% dataType:
% 1 means matrix form storage
% 2 means "x y value" form storage
if datasetId <= 6
    dataType = 1;
    prefix = '../../20-newsgroup/';
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../../Reuter/';
elseif datasetId == 10
    dataType = 2;
    prefix = '../../Animal_img/';
end

sizeOfOneFold = numSampleData/ numFold;

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

sampledSourceDataIndex = randperm(numSourceData, numSampleData);
sampledTargetDataIndex = randperm(numTargetData, numSampleData);
sourceDomainData = sourceDomainData(sampledSourceDataIndex, :);
targetDomainData = targetDomainData(sampledTargetDataIndex, :);
numSourceData = numSampleData;
numTargetData = numSampleData;

sourceY = sourceY(sampledSourceDataIndex);
targetY = targetY(sampledTargetDataIndex);
Y = [sourceY; targetY];

numAllData = numSourceData + numTargetData;

% Pre-allocate matrix K, L, and compute H
K = zeros(numAllData, numAllData);
L = zeros(numAllData, numAllData);
H = eye(numAllData) - ((1/(numAllData) * ones(numAllData, numAllData)));

for tuneMu = -6:6
    mu = 10 ^ tuneMu;
    numCorrectPredict = 0;
    for fold = 0: (numFold-1)
        fprintf('fold: %d\n', fold);
        % Compute K, L matrix
        testDataIndex = (fold*sizeOfOneFold+1: fold*sizeOfOneFold+sizeOfOneFold) + numSampleData;
        trainDataIndex = setdiff(1:numAllData, testDataIndex);
        trainY = Y(trainDataIndex);
        testY = Y(testDataIndex);
        
        for i = 1:numAllData
            for j = 1:numAllData
                if i > numSourceData
                    instance1 = targetDomainData(i-numSourceData, :);
                else
                    instance1 = sourceDomainData(i, :);
                end
                
                if j > numSourceData
                    instance2 = targetDomainData(j-numSourceData, :);
                else
                    instance2 = sourceDomainData(j, :);
                end
                
                K(i, j) = instance1 * instance2';
                
                if i < numSourceData && j < numSourceData
                    L(i, j) = 1/ numSourceData^2;
                elseif i > numSourceData && j > numSourceData
                    L(i, j) = 1/ numTargetData^2;
                else
                    L(i, j) = 1/ (numSourceData*numTargetData);
                end
            end
        end
        tcaMatrix = (K*L*K + mu*eye(numAllData))\(K*H*K);
        [eigVectorMatrix, eigValueMatrix] = eig(tcaMatrix);
        transformMatrix = eigVectorMatrix(:, 1:featureDimAfterReduce);
        % Project data in kernel space to the learned components
        dimReducedMatrix = tcaMatrix * transformMatrix;
        svmModel = fitcsvm(dimReducedMatrix(trainDataIndex, :), trainY, 'KernelFunction', 'linear');
        predictLabel = predict(svmModel, dimReducedMatrix(testDataIndex, :));
        
        for i = 1: sizeOfOneFold
            if(predictLabel(i) == testY(i))
                numCorrectPredict = numCorrectPredict + 1;
            end
        end
    end
    
    disp(numCorrectPredict/ numSampleData);
end