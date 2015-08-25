clear;
clc;
% if matlabpool('size') > 0
%     matlabpool close;
% end
% matlabpool('open', 'local', 4);

% configuration
exp_title = 'Motar2';
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isURandom = true;
datasetId = 1;
numSampleInstance = 500;
numSampleFeature = 2000;
maxIter = 100;
randomTryTime = 40;

if datasetId <= 6
    prefix = '../20-newsgroup/';
else
    prefix = '../Reuter/';
end
numDom = 2;
%sourceDomain = 1;
targetDomain = 2;

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

numSourceInstanceList = [3913 3907 3783 3954 3830 3823 1237 1016 897 5000 5000 5000 5000 5000 5000 5000];
numTargetInstanceList = [3925 3910 3336 3961 3387 3371 1207 1043 897 5000 5000 5000 5000 5000 5000 5000];
numSourceFeatureList = [57312 59470 60800 58470 60800 60800 4771 4415 4563 10940 2688 2000 252 2000 2000 2000];
numTargetFeatureList = [57914 59474 61188 59474 61188 61188 4771 4415 4563 10940];

numInstance = [numSourceInstanceList(datasetId) numTargetInstanceList(datasetId)];
numFeature = [numSourceFeatureList(datasetId) numTargetFeatureList(datasetId)];
numClass = 2;
numInstanceCluster = [3 3];
numFeatureCluster = [5 5];

sigma = 40;
alpha = 0;
beta = 0;
numCVFold = 5;
CVFoldSize = numSampleInstance/ numCVFold;

showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);

YTrue = cell(1, numDom);
Y = cell(1, numDom);
W = cell(1, numDom);
uc = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);
label = cell(1, numDom);

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, 1);

for i = 1: numDom
    %Randomly sample instances & the corresponding labels
    if isSampleInstance == true
        sampleInstanceIndex = randperm(numInstance(i), numSampleInstance);
        X{i} = X{i}(sampleInstanceIndex, :);
        numInstance(i) = numSampleInstance;
    end
    
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature);
        X{i} = X{i}(:, denseFeatures);
        numFeature(i) = numSampleFeature;
    end
end

parfor dom = 1: numDom
    Su{dom} = zeros(numInstance(dom), numInstance(dom));
    Su{dom} = gaussianSimilarityMatrix(X{dom}, sigma);
end