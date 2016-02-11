if datasetId <= 6
    dataType = 1;
    prefix = '../../../20-newsgroup/'; 
elseif datasetId > 6 && datasetId <= 9
    dataType = 1;
    prefix = '../../../Reuter/';
elseif datasetId >= 10 && datasteId <=13
    dataType = 2;
    prefix = '../../../Animal_img/';
end

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

TrueYMatrix = cell(1, numDom);
YMatrix = cell(1, numDom);
Label = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);
Sv = cell(1, numDom);
Dv = cell(1, numDom);
Lv = cell(1, numDom);

initV = cell(randomTryTime, numDom);
initU = cell(randomTryTime, numDom);
initH = cell(randomTryTime);

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);

sampleSourceDataIndex = csvread(sprintf('../../sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
sampleValidateDataIndex = csvread(sprintf('../../sampleIndex/sampleValidateDataIndex%d.csv', datasetId));
sampleTestDataIndex = csvread(sprintf('../../sampleIndex/sampleTargetDataIndex%d.csv', datasetId));

for i = 1: numDom
    domainName = domainNameList{i};
    Label{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
    %Randomly sample instances & the corresponding labels
%     fprintf('Sample domain %d data\n', i);
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
    TrueYMatrix{i} = zeros(numSampleInstance(i), numClass(i));
    for j = 1: numSampleInstance(i)        
        TrueYMatrix{i}(j, Label{i}(j)) = 1;
    end
    X{i} = normr(X{i});
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
    for user = 1:numSampleInstance(dom)
        Du{dom}(user,user) = sum(Su{dom}(user,:));
    end
    Lu{dom} = Du{dom} - Su{dom};
end

for dom = 1: numDom
    Sv{dom} = zeros(numSampleFeature(i), numSampleFeature(i));
    Dv{dom} = zeros(numSampleFeature(i), numSampleFeature(i));
    Lv{dom} = zeros(numSampleFeature(i), numSampleFeature(i));
    
    %user
    fprintf('Domain%d: calculating Sv, Dv, Lv\n', dom);
    Sv{dom} = gaussianSimilarityMatrix(X{dom}', sigma);
    Sv{dom}(isnan(Su{dom})) = 0;
    Sv{dom}(~isfinite(Su{dom})) = 1;
    for feature = 1:numSampleFeature(i)
        Dv{dom}(feature,feature) = sum(Sv{dom}(feature,:));
    end
    Lv{dom} = Dv{dom} - Sv{dom};
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
        [initU(t,:),initH{t},initV(t,:)] = randomInitialize(numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, numDom, isUsingTensor);
    end
end

CVFoldSize = numSampleInstance(targetDomain)/ numCVFold;