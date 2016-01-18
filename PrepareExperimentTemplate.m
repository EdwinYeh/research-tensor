clear;
clc;
% if matlabpool('size') > 0
%     matlabpool close;
% end
% matlabpool(4);

%configuration
datasetId = 1;
exp_title = sprintf('Motar2_W_%d', datasetId);
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isRandom = true;
numSampleInstance = [0, 0];
numSampleFeature = 20;
maxIter = 120;
randomTryTime = 1;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = [2, 2];
    sigma = 0.1;
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = [2, 2];
    sigma = 0.1;
elseif datasetId == 10
    dataType = 2;
    prefix = '../Animal_img/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = [2, 2];
    sigma = 0.1;
end
numDom = 2;
sourceDomain = 1;
targetDomain = 2;

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

numSourceInstanceList = [3913 3906 3782 3953 3829 3822 1237 1016 897 4460 60];
numTargetInstanceList = [3925 3909 3338 3960 3389 3373 1207 1043 897 4601 30];
numSourceFeatureList = [57309 59463 60800 58463 60800 60800 4771 4415 4563 4940 26];
numTargetFeatureList = [57913 59474 61188 59474 61188 61188 4771 4415 4563 4940 26];

alpha = 0;
beta = 0;

YMatrix = cell(1, numDom);
Label = cell(1, numDom);
uc = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);

initV = cell(randomTryTime, numDom);
initU = cell(randomTryTime, numDom);
initB = cell(randomTryTime);

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);

sampleSourceDataIndex = csvread(sprintf('%ssampleSourceIndex%d.csv', prefix, datasetId));
sampleTargetDataIndex = csvread(sprintf('%ssampleTestIndex%d.csv', prefix, datasetId));

for i = 1: numDom
    domainName = domainNameList{i};
    Label{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
    X{i} = normr(X{i});
    %Randomly sample instances & the corresponding labels
    if isSampleInstance == true
        if i == sourceDomain
            X{i} = X{i}(sampleSourceDataIndex, :);
            Label{i} = Label{i}(sampleSourceDataIndex, :);
        elseif i == targetDomain
            X{i} = X{i}(sampleTargetDataIndex, :);
            Label{i} = Label{i}(sampleTargetDataIndex, :);
        end
    end
    [numSampleInstance(i), ~] = size(X{i});
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature);
        X{i} = X{i}(:, denseFeatures);
    end
    YMatrix{i} = zeros(numSampleInstance(i), numClass(i));
    for j = 1: numSampleInstance(i)        
        YMatrix{i}(j, Label{i}(j)) = 1;
    end
end

for dom = 1: numDom
    W{dom} = zeros(numSampleInstance(dom), numClass(dom));
    Su{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    Du{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    Lu{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    
    %user
    fprintf('Domain%d: calculating Su, Du, Lu\n', dom);
    Su{dom} = gaussianSimilarityMatrix(X{dom}, sigma);
    for useri = 1:numSampleInstance(dom)
        Du{dom}(useri,useri) = sum(Su{dom}(useri,:));
    end
    Lu{dom} = Du{dom} - Su{dom};
end

str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster, numFeatureCluster);
end
str = str(1:length(str)-1);
eval(sprintf('originalSize = [%s];', str));

%Randomly initialize B, U, V
if isRandom == true
    for t = 1: randomTryTime
        [initU(t,:),initB{t},initV(t,:)] = randomInitialize(numSampleInstance, numClass, numInstanceCluster, numFeatureCluster, numDom, true);
    end
end

numCVFold = 5;
CVFoldSize = numSampleInstance(targetDomain)/ numCVFold;

showExperimentInfo(datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);