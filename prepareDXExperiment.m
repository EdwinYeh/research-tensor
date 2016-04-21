% configuration

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

numSampleInstance = [0 0];

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
    if isSampleInstance == true
        if dom == sourceDomain
            X{dom} = X{dom}(sampleSourceDataIndex, :);
            sampledLabel{dom} = allLabel{dom}(sampleSourceDataIndex, :);
        elseif dom == targetDomain
            X{dom} = X{dom}(sampleTargetDataIndex, :);
            sampledLabel{dom} = allLabel{dom}(sampleTargetDataIndex, :);
        end
    end
     [numSampleInstance(dom), ~] = size(X{dom});
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{dom}, numSampleFeature);
        X{dom} = X{dom}(:, denseFeatures);
    end
end

for dom = 1: numDom
    Su{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    Du{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    Lu{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    Sv{dom} = zeros(numSampleFeature, numSampleFeature);
    Dv{dom} = zeros(numSampleFeature, numSampleFeature);
    Lv{dom} = zeros(numSampleFeature, numSampleFeature);
    
    %user
    fprintf('Domain%d: calculating Su, Du, Lu\n', dom);
    for useri = 1:numSampleInstance(dom)
        for userj = 1:numSampleInstance(dom)
            %ndsparse does not support norm()
            dif = norm((X{dom}(useri, :) - X{dom}(userj,:)));
            Su{dom}(useri, userj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for useri = 1:numSampleInstance(dom)
        Du{dom}(useri,useri) = sum(Su{dom}(useri,:));
    end
    Lu{dom} = Du{dom} - Su{dom};
    %item
    fprintf('Domain%d: calculating Sv, Dv, Lv\n', dom);
    for itemi = 1:numSampleFeature
        for itemj = 1:numSampleFeature
            %ndsparse does not support norm()
            dif = norm((X{dom}(:,itemi) - X{dom}(:,itemj)));
            Sv{dom}(itemi, itemj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for itemi = 1:numSampleFeature
        Dv{dom}(itemi,itemi) = sum(Sv{dom}(itemi,:));
    end
    Lv{dom} = Dv{dom} - Sv{dom};
end

numCVFold = 5;
CVFoldSize = numSampleInstance(targetDomain)/ numCVFold;