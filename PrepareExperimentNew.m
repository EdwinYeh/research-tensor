if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
elseif datasetId > 6 && datasetId <= 9
    dataType = 1;
    prefix = '../Reuter/';
elseif datasetId >= 10
    dataType = 2;
    prefix = '../Animal_img/';
end

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

TrueYMatrix = cell(1, numDom);
YMatrix = cell(1, numDom);
Label = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);
XTrain = cell(1,numDom);
XTest = cell(1,numDom);
LabelTrain = cell(1, numDom);
LabelTest = cell(1, numDom);
YTrain = cell(1, numDom);
YTest = cell(1, numDom);
numTrainData = zeros(1, numDom);
numTestData = zeros(1, numDom);

sourceTrainIndex = csvread(sprintf('sampleIndex/%s/sampleSourceTrainIndex%d.csv', sampleSizeLevel, datasetId));
sourceTestIndex = csvread(sprintf('sampleIndex/%s/sampleSourceTestIndex%d.csv', sampleSizeLevel, datasetId));
targetTrainIndex = csvread(sprintf('sampleIndex/%s/sampleTargetTrainIndex%d.csv', sampleSizeLevel, datasetId));
targetTestIndex = csvread(sprintf('sampleIndex/%s/sampleTargetTestIndex%d.csv', sampleSizeLevel, datasetId));

for i = 1: numDom
    domainName = domainNameList{i};
    Label{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
    %Randomly sample instances & the corresponding labels
    %     fprintf('Sample domain %d data\n', i);
    if isSampleInstance == true
        if i == sourceDomain
            XTest{i} = X{i}(sourceTestIndex, :);
            XTrain{i} = X{i}(sourceTrainIndex, :);
            LabelTest{i} = Label{i}(sourceTestIndex, :);
            LabelTrain{i} = Label{i}(sourceTrainIndex, :);
        elseif i == targetDomain
            XTest{i} = X{i}(targetTestIndex, :);
            XTrain{i} = X{i}(targetTrainIndex, :);
            LabelTest{i} = Label{i}(targetTestIndex, :);
            LabelTrain{i} = Label{i}(targetTrainIndex, :);
        end
    end
    [numTrainData(i), ~] = size(XTrain{i});
    [numTestData(i), ~] = size(XTest{i});
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature);
        XTrain{i} = XTrain{i}(:, denseFeatures);
        XTest{i} = XTest{i}(:, denseFeatures);
    end
    XTrain{i} = normr(XTrain{i});
    XTest{i} = normr(XTest{i});
    
    YTrain{i} = zeros(numTrainData(i), numClass(i));
    YTest{i} = zeros(numTestData(i), numClass(i));
    for j = 1: numTrainData(i)
        YTrain{i}(j, LabelTrain{i}(j)) = 1;
    end 
    for j = 1: numTestData(i)
        YTest{i}(j, LabelTest{i}(j)) = 1;
    end
end

prepareTimer = tic;
for dom = 1: numDom
    Su{dom} = zeros(numTrainData(dom), numTrainData(dom));
    Du{dom} = zeros(numTrainData(dom), numTrainData(dom));
%     Lu{dom} = zeros(numTrainData(dom), numTrainData(dom));
    
    %user
    %     fprintf('Domain%d: calculating Su, Du, Lu\n', dom);
    Su{dom} = gaussianSimilarityMatrix(XTrain{dom}, sigma);
    Su{dom}(isnan(Su{dom})) = 0;
    for useri = 1:numTrainData(dom)
        Du{dom}(useri,useri) = sum(Su{dom}(useri,:));
    end
%     Lu{dom} = Du{dom} - Su{dom};
end

prepareTime = toc(prepareTimer);
disp(prepareTime);
