% configuration
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isRandom = true;
numSampleInstance = [500, 500];
numSampleFeature = [2000, 2000];
maxIter = 500;
randomTryTime = 5;
cpRank = 5;
numClass = 2;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
elseif datasetId >= 10 && datasetId <= 13
    dataType = 2;
    prefix = '../Animal_img/';
end
numDom = 2;
sourceDomain = 1;
targetDomain = 2;

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);
sampleSourceDataIndex = csvread(sprintf('sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
sampleTargetDataIndex = csvread(sprintf('sampleIndex/sampleTargetDataIndex%d.csv', datasetId));

numInstance = [size(X{1}, 1) size(X{2}, 1)];
numFeature = [size(X{1}, 2) size(X{2}, 2)];

alpha = 0;
beta = 0;

Sv = cell(1, numDom);
Dv = cell(1, numDom);
Lv = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);
allLabel = cell(1, numDom);
sampledLabel = cell(1, numDom);

for dom = 1:numDom
    domainName = domainNameList{dom};
    allLabel{dom} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
end

for dom = 1: numDom
    X{dom} = normr(X{dom});
    if dom == sourceDomain
        sampleDataIndex = sampleSourceDataIndex;
    elseif dom == targetDomain
        sampleDataIndex = sampleTargetDataIndex;
    end
    if isSampleInstance == true
        X{dom} = X{dom}(sampleDataIndex, :);
        numInstance(dom) = numSampleInstance(dom);
        sampledLabel{dom} = allLabel{dom}(sampleDataIndex, :);
    end
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{dom}, numSampleFeature(dom));
        X{dom} = X{dom}(:, denseFeatures);
        numFeature(dom) = numSampleFeature(dom);
    end
end

for dom = 1: numDom
    Su{dom} = zeros(numInstance(dom), numInstance(dom));
    Du{dom} = zeros(numInstance(dom), numInstance(dom));
    Lu{dom} = zeros(numInstance(dom), numInstance(dom));
    Sv{dom} = zeros(numFeature(dom), numFeature(dom));
    Dv{dom} = zeros(numFeature(dom), numFeature(dom));
    Lv{dom} = zeros(numFeature(dom), numFeature(dom));
    
    %user
    fprintf('Domain%d: calculating Su, Du, Lu\n', dom);
    for useri = 1:numInstance(dom)
        for userj = 1:numInstance(dom)
            %ndsparse does not support norm()
            dif = norm((X{dom}(useri, :) - X{dom}(userj,:)));
            Su{dom}(useri, userj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for useri = 1:numInstance(dom)
        Du{dom}(useri,useri) = sum(Su{dom}(useri,:));
    end
    Lu{dom} = Du{dom} - Su{dom};
    %item
    fprintf('Domain%d: calculating Sv, Dv, Lv\n', dom);
    for itemi = 1:numFeature(dom)
        for itemj = 1:numFeature(dom)
            %ndsparse does not support norm()
            dif = norm((X{dom}(:,itemi) - X{dom}(:,itemj)));
            Sv{dom}(itemi, itemj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for itemi = 1:numFeature(dom)
        Dv{dom}(itemi,itemi) = sum(Sv{dom}(itemi,:));
    end
    Lv{dom} = Dv{dom} - Sv{dom};
end

%initialize B, U, V
initV = cell(randomTryTime, numDom);
initU = cell(randomTryTime, numDom);
initB = cell(randomTryTime);

str = '';
for dom = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster, numFeatureCluster);
end
str = str(1:length(str)-1);
for t = 1: randomTryTime
    for dom = 1: numDom
        initU{t,dom} = rand(numInstance(dom), numInstanceCluster);
        initV{t,dom} = rand(numFeature(dom), numFeatureCluster);
    end
    randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
    randStr = round(randStr);
    initB{t} = tensor(randStr);
end

CP1 = rand(numInstanceCluster, cpRank);
CP2 = rand(numFeatureCluster, cpRank);
CP3 = rand(numInstanceCluster, cpRank);
CP4 = rand(numFeatureCluster, cpRank);

numCVFold = 5;
CVFoldSize = numInstance(targetDomain)/ numCVFold;