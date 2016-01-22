% if matlabpool('size') > 0
%     matlabpool close;
% end
% matlabpool(4);

TrueYMatrix = cell(1, numDom);
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
sampleTargetDataIndex = csvread(sprintf('%ssampleTargetIndex%d.csv', prefix, datasetId));
% 
for i = 1: numDom
    domainName = domainNameList{i};
    Label{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
    X{i} = minMaxNormalize(X{i});
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
    TrueYMatrix{i} = zeros(numSampleInstance(i), numClass(i));
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

numCVFold = 5;
CVFoldSize = numSampleInstance(targetDomain)/ numCVFold;