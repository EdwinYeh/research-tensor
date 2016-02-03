% if matlabpool('size') > 0
%     matlabpool close;
% end
% matlabpool(4);

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/'; 
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
elseif datasetId == 10
    dataType = 2;
    prefix = '../Animal_img/';
elseif datasetId == 11 || datasetId == 12;
    dataType = 2;
    prefix = '../Toy_dataset/';
end

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

TrueYMatrix = cell(1, numDom);
YMatrix = cell(1, numDom);
Label = cell(1, numDom);
uc = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);

initV = cell(randomTryTime, numDom);
initU = cell(randomTryTime, numDom);
initB = cell(randomTryTime);

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);

sampleSourceDataIndex = csvread(sprintf('sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
sampleValidateDataIndex = csvread(sprintf('sampleIndex/sampleValidateDataIndex%d.csv', datasetId));
sampleTestDataIndex = csvread(sprintf('sampleIndex/sampleTargetDataIndex%d.csv', datasetId));
% [numSourceInstance, ~] = size(X{sourceDomain});
% [numTargetInstance, ~] = size(X{targetDomain});
% sampleSourceDataIndex = randperm(numSourceInstance, numSampleInstance(sourceDomain));
% sampleTargetDataIndex = randperm(numTargetInstance, numSampleInstance(targetDomain));
% 
for i = 1: numDom
    domainName = domainNameList{i};
    Label{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
    %Randomly sample instances & the corresponding labels
    if isSampleInstance == true
        if i == sourceDomain
            X{i} = X{i}(sampleSourceDataIndex, :);
            Label{i} = Label{i}(sampleSourceDataIndex, :);
        elseif i == targetDomain
            if isTestPhase == true
                X{i} = X{i}(sampleTestDataIndex, :);
                Label{i} = Label{i}(sampleTestDataIndex, :);
            else
                X{i} = X{i}(sampleValidateDataIndex, :);
                Label{i} = Label{i}(sampleValidateDataIndex, :);
            end
        end
    end
    [numSampleInstance(i), ~] = size(X{i});
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature);
        X{i} = X{i}(:, denseFeatures);
    end
    X{i} = normr(X{i});
    TrueYMatrix{i} = -1* ones(numSampleInstance(i), numClass(i));
    for j = 1: numSampleInstance(i)        
        TrueYMatrix{i}(j, Label{i}(j)) = 1;
    end
end
% 
for dom = 1: numDom
    Su{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    Du{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    Lu{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    
    %user
    fprintf('Domain%d: calculating Su, Du, Lu\n', dom);
    Su{dom} = gaussianSimilarityMatrix(X{dom}, sigma);
    Su{dom}(isnan(Su{dom})) = 0;
    Su{dom}(~isfinite(Su{dom})) = 1;
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
        [initU(t,:),initB{t},initV(t,:)] = randomInitialize(numSampleInstance, numClass, numInstanceCluster, numFeatureCluster, numDom, isUsingTensor);
    end
end

CVFoldSize = numSampleInstance(targetDomain)/ numCVFold;