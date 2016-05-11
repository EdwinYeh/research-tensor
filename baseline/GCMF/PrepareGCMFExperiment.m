if datasetId <= 6
    dataType = 1;
    prefix = '../../../20-newsgroup/';
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../../../Reuter/';
elseif datasetId >= 10 && datasetId <= 13
    dataType = 2;
    prefix = '../../../Animal_img/';
end
numDom = 2;
sourceDomain = 1;
targetDomain = 2;

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);
sourceDataIndex = csvread(sprintf('../../sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
validationDataIndex = csvread(sprintf('../../sampleIndex/sampleValidationDataIndex%d.csv', datasetId));
testDataIndex = csvread(sprintf('../../sampleIndex/sampleTestDataIndex%d.csv', datasetId));
disp(length(validationDataIndex))

numSampleInstance = [0 0];

Sv = cell(1, numDom);
Dv = cell(1, numDom);
Lv = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);
TrueYMatrix = cell(1, numDom);
allLabel = cell(1, numDom);
sampleLabel = cell(1, numDom);

for dom = 1: numDom
    domainName = domainNameList{dom};
    allLabel{dom} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
    if isSampleInstance == true
        if dom == sourceDomain
            X{dom} = X{dom}(sourceDataIndex, :);
            sampleLabel{dom} = allLabel{dom}(sourceDataIndex, :);
        elseif dom == targetDomain
            if isTestPhase == true
                X{dom} = [X{dom}(validationDataIndex, :); X{dom}(testDataIndex, :)];
                sampleLabel{dom} = [allLabel{dom}(validationDataIndex, :); allLabel{dom}(testDataIndex)];
            else
                X{dom} = X{dom}(validationDataIndex, :);
                sampleLabel{dom} = allLabel{dom}(validationDataIndex, :);
            end
        end
    end
    [numSampleInstance(dom), ~] = size(X{dom});
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{dom}, numSampleFeature);
        X{dom} = X{dom}(:, denseFeatures);
    end
    TrueYMatrix{dom} = zeros(numSampleInstance(dom), numClass(dom));
    for j = 1: numSampleInstance(dom)
        TrueYMatrix{dom}(j, sampleLabel{dom}(j)) = 1;
    end
    X{dom} = normr(X{dom});
end

numValidationInstance = length(validationDataIndex);
numTestInstance = length(testDataIndex);



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
            Sv{dom}(itemi, itemj) = exp(-(dif*dif)/(2*sigma2));
        end
    end
    for itemi = 1:numSampleFeature
        Dv{dom}(itemi,itemi) = sum(Sv{dom}(itemi,:));
    end
    Lv{dom} = Dv{dom} - Sv{dom};
end

if ~isTestPhase
    CVFoldSize = numSampleInstance(targetDomain)/ numCVFold;
else
    CVFoldSize = numTestInstance/ numCVFold;
end