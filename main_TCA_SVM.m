datasetId = 1;

% dataType:
% 1 means matrix form storage
% 2 means "x y value" form storage
if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
elseif datasetId == 10
    dataType = 2;
    prefix = '../Animal_img/';
end

numDom = 2;

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

% Load data from source and target domain data
X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);
for i = 1:numDom
    denseFeatureIndex = findDenseFeature(X{i}, 4000);
    X{i} = X{i}(:, denseFeatureIndex);
end
sourceDomainData = X{1};
targetDomainData = X{2};
sizeOfSourceData = size(sourceDomainData);
sizeOfTargetData = size(targetDomainData);
numSourceInstance = sizeOfSourceData(1);
numTargetInstance = sizeOfTargetData(1);

sampledSourceInstanceIndex = randperm(numSourceInstance, 50);
sampledTargetInstanceIndex = randperm(numTargetInstance, 50);
numSourceInstance = 50;
numTargetInstance = 50;

numAllInstance = numSourceInstance + numTargetInstance;

% Pre-allocate matrix K, L, and compute H
K = zeros(numAllInstance, numAllInstance);
L = zeros(numAllInstance, numAllInstance);
H = eye(numAllInstance) - ((1/(numAllInstance) * ones(numAllInstance, numAllInstance)));

% Compute K, L matrix
for i = 1:numAllInstance
    for j = 1:numAllInstance
        if i > numSourceInstance
            instance1 = targetDomainData(i - numSourceInstance, :);
        else
            instance1 = sourceDomainData(i, :);
        end
        
        if j > numSourceInstance
            instance2 = targetDomainData(j - numSourceInstance, :);
        else
            instance2 = sourceDomainData(j, :);
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

mu = 1;
tcaMatrix = (K*L*K + mu*eye(numAllInstance))\(K*H*K);
[eigVectorMatrix, eigValueMatrix] = eig(tcaMatrix);
transformMatrix(:, 30);

