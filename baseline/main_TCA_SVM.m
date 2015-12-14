datasetId = 1;
numDom = 2;
mu = 1;
numSampleFeature = 2000;
numSampleInstance = 50;
numFold = 5;
featureDimAfterReduce = 30;

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

sizeOfOneFold = numSampleInstance/ numFold;

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
sizeOfSourceData = size(sourceDomainData);
sizeOfTargetData = size(targetDomainData);
numSourceInstance = sizeOfSourceData(1);
numTargetInstance = sizeOfTargetData(1);

sampledSourceInstanceIndex = randperm(numSourceInstance, numSampleInstance);
sampledTargetInstanceIndex = randperm(numTargetInstance, numSampleInstance);
numSourceInstance = numSampleInstance;
% Hide 20% data in target domain to be the test set
numTargetInstance = numSampleInstance - sizeOfOneFold;

numAllInstance = numSourceInstance + numTargetInstance;

% Pre-allocate matrix K, L, and compute H
K = zeros(numAllInstance, numAllInstance);
L = zeros(numAllInstance, numAllInstance);
H = eye(numAllInstance) - ((1/(numAllInstance) * ones(numAllInstance, numAllInstance)));

for fold = 0: (numFold-1)
    fold
    % Compute K, L matrix
    testDataIndex = sampledTargetInstanceIndex(fold*sizeOfOneFold+1: fold*sizeOfOneFold+sizeOfOneFold);
    sampledTargetInstanceIndex = setdiff(sampledTargetInstanceIndex, testDataIndex);
    trainY = [sourceY(sampledSourceInstanceIndex); targetY(sampledTargetInstanceIndex)];
    testY = targetY(testDataIndex);
    
    for i = 1:numAllInstance
        for j = 1:numAllInstance
            if i > numSourceInstance
                instance1 = targetDomainData(sampledTargetInstanceIndex(i-numSourceInstance), :);
            else
                instance1 = sourceDomainData(sampledSourceInstanceIndex(i), :);
            end
            
            if j > numSourceInstance
                instance2 = targetDomainData(sampledTargetInstanceIndex(j-numSourceInstance), :);
            else
                instance2 = sourceDomainData(sampledSourceInstanceIndex(j), :);
            end
            
            K(i, j) = instance1 * instance2';
            
            if i < numSourceInstance && j < numSourceInstance
                L(i, j) = 1/ numSourceInstance^2;
            elseif i > numSourceInstance && j > numSourceInstance
                L(i, j) = 1/ numTargetInstance^2;
            else
                L(i, j) = 1/ (numSourceInstance*numTargetInstance);
            end
        end
    end
    tcaMatrix = (K*L*K + mu*eye(numAllInstance))\(K*H*K);
    [eigVectorMatrix, eigValueMatrix] = eig(tcaMatrix);
    transformMatrix = eigVectorMatrix(:, featureDimAfterReduce);
    % Project data in kernel space to the learned components
    dimReducedMatrix = tcaMatrix * transformMatrix;
    svmModel = fitcsvm(dimReducedMatrix, trainY, 'CrossVal', 'on', 'KernelFunction', 'linear');
    predictLabel = svmModel.predict(targetDomainData(testDataIndex));
    predictLabel
end
