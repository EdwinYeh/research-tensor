SetParameter;
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
resultDirectory = sprintf('../exp_result/SVM/%d/', datasetId);
mkdir(resultDirectory);
expTitle = sprintf('SVM_%d', datasetId);
sampleSizeLevel = '1000_100';
resultFile = fopen(sprintf('%s%s.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'accuracy1,accuracy2\n');

CVFoldSize = numTrainData(1)/CVFoldNum;
avgValidationAccuracy = zeros(numDom, 1);
trainingTime = 0;
validationIndex = 1:CVFoldSize;
for cvFold = 1:CVFoldNum
    % S: selection matrix for validation set
    for domID = 1:length(X)
        % Calculate training index
        trainIndex = setdiff(1:numTrainData(domID), validationIndex);
        fprintf('validation index domain%d: %d~%d\n', domID, min(validationIndex), max(validationIndex))
        % XTrain/YTrain here means all data
        input.X{domID} = XTrain{domID}(trainIndex, :);
        input.Y{domID} = YTrain{domID}(trainIndex, :);
    end;
    
    validationAccuracy = zeros(1, numDom);
    for domID = 1:numDom
        SVMModel = fitcsvm(XTrain{domID}(trainIndex,:),LabelTrain{domID}(trainIndex,:),...
            'Standardize',true,'KernelFunction','RBF',...
            'KernelScale','auto');
        validationLabel = predict(SVMModel, XTrain{domID}(validationIndex, :));
        validationAccuracy(domID) =  0;
        for i  = 1:length(validationIndex)
            if validationLabel(i) == YTrain{domID}(i)
                validationAccuracy(domID) = validationAccuracy(domID) + 1;
            end
        end
        validationAccuracy(domID) = validationAccuracy(domID)/length(validationIndex);
        avgValidationAccuracy(domID) = avgValidationAccuracy(domID) + validationAccuracy(domID);
    end
    disp(validationAccuracy);
    % disp(validationAccuracy);
    validationIndex = validationIndex + CVFoldSize;
end
avgValidationAccuracy = avgValidationAccuracy/(CVFoldNum);
fprintf(resultFile, '%g,%g\n', avgValidationAccuracy(1), avgValidationAccuracy(2));
fclose(resultFile);
