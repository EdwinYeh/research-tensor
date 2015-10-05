clear;
clc;
if matlabpool('size') > 0
    matlabpool close;
end
matlabpool(4);

% configuration
exp_title = 'Motar2_W_10';
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isRandom = true;
datasetId = 10;
numSampleInstance = 500;
numSampleFeature = 2000;
maxIter = 100;
randomTryTime = 20;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
elseif datasetId == 10
    dataType = 2;
    prefix = '../Animal_img/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
elseif datasetId == 11
    dataType = 1;
    prefix = '../song/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
end
numDom = 2;
%sourceDomain = 1;
targetDomain = 2;

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

numSourceInstanceList = [3913 3906 3782 3953 3829 3822 1237 1016 897 4460 229];
numTargetInstanceList = [3925 3909 3338 3960 3389 3373 1207 1043 897 4601 223];
numSourceFeatureList = [57309 59463 60800 58463 60800 60800 4771 4415 4563 4940 10783];
numTargetFeatureList = [57913 59474 61188 59474 61188 61188 4771 4415 4563 4940 10783];

numInstance = [numSourceInstanceList(datasetId) numTargetInstanceList(datasetId)];
numFeature = [numSourceFeatureList(datasetId) numTargetFeatureList(datasetId)];

alpha = 0;
beta = 0;
numCVFold = 5;
CVFoldSize = numSampleInstance/ numCVFold;

showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);

% disp(numSampleFeature);
%disp(sprintf('Configuration:\n\tisUpdateAE:%d\n\tisUpdateFi:%d\n\tisBinary:%d\n\tmaxIter:%d\n\t#domain:%d (predict domain:%d)', isUpdateAE, isUpdateFi, isBinary, maxIter, numDom, targetDomain));
%disp(sprintf('#users:[%s]\n#items:[%s]\n#user_cluster:[%s]\n#item_cluster:[%s]', num2str(numInstance(1:numDom)), num2str(numFeature(1:numDom)), num2str(numInstanceCluster(1:numDom)), num2str(numFeatureCluster(1:numDom))));

%[groundTruthX, snapshot, idx] = preprocessing(numDom, targetDomain);
%bestLambda = 0.1;
%bestAccuracy = 0;

%Bcell = cell(1, numDom);
YTrue = cell(1, numDom);
Y = cell(1, numDom);
W = cell(1, numDom);
uc = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);
label = cell(1, numDom);

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);
% X = load(sprintf('%sdataset%d.mat', prefix, datasetId));
% X = X.X;

for i = 1:numDom
    domainName = domainNameList{i};
    label{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
end

for i = 1: numDom
    X{i} = normr(X{i});
    %Randomly sample instances & the corresponding labels
    if isSampleInstance == true
        sampleInstanceIndex = randperm(numInstance(i), numSampleInstance);
        X{i} = X{i}(sampleInstanceIndex, :);
        numInstance(i) = numSampleInstance;
        label{i} = label{i}(sampleInstanceIndex, :);
    end
    YTrue{i} = zeros(numInstance(i), numClass);
    for j = 1: numInstance(i)        
        YTrue{i}(j, label{i}(j)) = 1;
    end
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature);
        X{i} = X{i}(:, denseFeatures);
        numFeature(i) = numSampleFeature;
    end
end

% disp('Train logistic regression');
% logisticCoefficient = glmfit(X{1}, label{1} - 1, 'binomial');

parfor dom = 1: numDom
    W{dom} = zeros(numInstance(dom), numClass);
    Su{dom} = zeros(numInstance(dom), numInstance(dom));
    Du{dom} = zeros(numInstance(dom), numInstance(dom));
    Lu{dom} = zeros(numInstance(dom), numInstance(dom));
    
    W{dom}(YTrue{dom}~=0) = 1;
    
    %user
    fprintf('Domain%d: calculating Su, Du, Lu\n', dom);
    Su{dom} = gaussianSimilarityMatrix(X{dom}, sigma);
    for useri = 1:numInstance(dom)
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

resultFile = fopen(sprintf('score_accuracy_%s.csv', exp_title), 'w');
disp('Start training')
%initialize B, U, V
initV = cell(randomTryTime, numDom);
initU = cell(randomTryTime, numDom);
initB = cell(randomTryTime);

if isRandom == true
    for t = 1: randomTryTime
        [initU(t,:),initB{t},initV(t,:)] = randomInitialize(numInstance, numClass, numInstanceCluster, numFeatureCluster, numDom, true);
    end
end

for tuneLambda = 0:6
    lambda = 0.000001 * 10 ^ tuneLambda;
    time = round(clock);
    fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
    fprintf('Use Lambda:%f\n', lambda);
    localBestAccuracy = 0;
    localBestScore = Inf;
    fileIsOpened = false;
    %each pair is (objective score, accuracy);
    resultCellArray = cell(randomTryTime, 2);
    parfor t = 1: randomTryTime
        validateScore = 0;
        validateIndex = 1: CVFoldSize;
        foldObjectiveScores = zeros(1,numCVFold);
        for fold = 1:numCVFold
            %Iterative update
%             fprintf('fold: %d\n', fold);
            U = initU(t, :);
            V = initV(t, :);
            B = initB{t};
            Y = YTrue;
            Y{targetDomain}(validateIndex, :) = 0;
            W = ones(numSampleInstance, numClass);
            W(validateIndex, :) = 0;
            iter = 0;
            diff = -1;
            newObjectiveScore = Inf;
            MAES = zeros(1,maxIter);
            RMSES = zeros(1,maxIter);
            
            while (abs(diff) >= 0.001  && iter < maxIter)
                iter = iter + 1;
                oldObjectiveScore = newObjectiveScore;
                %                         fprintf('\t#Iterator:%d', iter);
                %                         disp([newObjectiveScore, diff]);
                newObjectiveScore = 0;
                for i = 1:numDom
%                     time = round(clock);
%                     fprintf('New iteration: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
                    %disp(sprintf('\tdomain #%d update...', i));
                    [projB, threeMatrixB] = SumOfMatricize(B, 2*(i - 1)+1);
                    %bestCPR = FindBestRank(threeMatrixB, 50)
                    bestCPR = 20;
                    CP = cp_apr(tensor(threeMatrixB), bestCPR, 'printitn', 0, 'alg', 'mu');%parafac_als(tensor(threeMatrixB), bestCPR);
%                     time = round(clock);
%                     fprintf('Cp complete: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
                    A = CP.U{1};
                    E = CP.U{2};
                    U3 = CP.U{3};

                    fi = cell(1, length(CP.U{3}));

                    %disp(sprintf('\t\tupdate V...'));
                    %update V
                    if i == targetDomain
                        V{i} = V{i}.*sqrt((Y{i}'*U{i}*projB)./((V{i}*projB'*U{i}'.*W')*U{i}*projB));
                    else
                        V{i} = V{i}.*sqrt((Y{i}'*U{i}*projB)./((V{i}*projB'*U{i}')*U{i}*projB));
                    end
                    V{i}(isnan(V{i})) = 0;
                    V{i}(~isfinite(V{i})) = 0;
                    %col normalize
                    [r, ~] = size(V{i});
                    for tmpI = 1:r
                        bot = sum(abs(V{i}(tmpI,:)));
                        if bot == 0
                            bot = 1;
                        end
                        V{i}(tmpI,:) = V{i}(tmpI,:)/bot;
                    end
                    V{i}(isnan(V{i})) = 0;
                    V{i}(~isfinite(V{i})) = 0;
%                     time = round(clock);
%                     fprintf('V updated: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
                    %disp(sprintf('\t\tupdate U...'));
                    %update U
                    if i == targetDomain
                        U{i} = U{i}.*sqrt((Y{i}*V{i}*projB' + lambda*Su{i}*U{i})./((U{i}*projB*V{i}'.*W)*V{i}*projB' + lambda*Du{i}*U{i}));
                    else
                        U{i} = U{i}.*sqrt((Y{i}*V{i}*projB' + lambda*Su{i}*U{i})./(U{i}*projB*V{i}'*V{i}*projB' + lambda*Du{i}*U{i}));
                    end
                    U{i}(isnan(U{i})) = 0;
                    U{i}(~isfinite(U{i})) = 0;
                    %col normalize
                    [r c] = size(U{i});
                    for tmpI = 1:r
                        bot = sum(abs(U{i}(tmpI,:)));
                        if bot == 0
                            bot = 1;
                        end
                        U{i}(tmpI,:) = U{i}(tmpI,:)/bot;
                    end
                    U{i}(isnan(U{i})) = 0;
                    U{i}(~isfinite(U{i})) = 0;
%                     time = round(clock);
%                     fprintf('U updated: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
                    %update fi
                    [r, c] = size(U3);
                    nextThreeB = zeros(numInstanceCluster, numFeatureCluster, r);
                    sumFi = zeros(c, c);
                    CPLamda = CP.lambda(:);
                    for idx = 1:r
                        %for idx = 1:r
                        fi{idx} = diag(CPLamda.*U3(idx,:)');
                        sumFi = sumFi + fi{idx};
                    end
                    if isUpdateAE
                        %disp(sprintf('\t\tupdate A...'));
                        [rA, cA] = size(A);
                        onesA = ones(rA, cA);
                        A = A.*sqrt((U{i}'*Y{i}*V{i}*E*sumFi + alpha*(onesA))./(U{i}'*U{i}*A*sumFi*E'*V{i}'*V{i}*E*sumFi));
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        %A = (spdiags (sum(abs(A),1)', 0, cA, cA)\A')';
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;

                        %disp(sprintf('\t\tupdate E...'));
                        [rE ,cE] = size(E);
                        onesE = ones(rE, cE);
                        E = E.*sqrt((V{i}'*Y{i}'*U{i}*A*sumFi + beta*(onesE))./(V{i}'*V{i}*E*sumFi*A'*U{i}'*U{i}*A*sumFi));
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        %E = (spdiags (sum(abs(E),1)', 0, cE, cE)\E')';
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;

                        %disp(sprintf('\tcombine next iterator B...'));
                        for idx = 1:r
                            nextThreeB(:,:,idx) = A*fi{idx}*E';
                        end
                    end
                    
                    B = InverseThreeToOriginalB(tensor(nextThreeB), 2*(i-1)+1, originalSize);
%                     time = round(clock);
%                     fprintf('AE updated: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
                end
                %disp(sprintf('\tCalculate this iterator error'));
                for i = 1:numDom
                    %for i = 1:numDom
                    [projB, ~] = SumOfMatricize(B, 2*(i - 1)+1);
                    result = U{i}*projB*V{i}';
                    if i == targetDomain
                        normEmp = norm((Y{i} - result).*W)*norm((Y{i} - result).*W);
                    else
                        normEmp = norm((Y{i} - result))*norm((Y{i} - result));
                    end
                    smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                    objectiveScore = normEmp + smoothU;
                    newObjectiveScore = newObjectiveScore + objectiveScore;
%                     fprintf('rank U: %d, rank V: %d\n', rank(U{i}), rank(V{i}));
                end
                %disp(sprintf('\tEmperical Error:%f', newObjectiveScore));
                %fprintf('iter:%d, error = %f\n', iter, newObjectiveScore);
                diff = oldObjectiveScore - newObjectiveScore;
%                 disp(diff);
%                 time = round(clock);
%                 fprintf('Objective score: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
            end
            foldObjectiveScores(fold) = newObjectiveScore;
%             fprintf('domain #%d => empTerm:%f, smoothU:%f ==> objective score:%f\n', i, normEmp, smoothU, objectiveScore);
%             fprintf('rank U: %d, rank V: %d\n', rank(U{1}), rank(V{1}));
%             fprintf('rank U: %d, rank V: %d\n', rank(U{2}), rank(V{2}));
            %calculate validationScore
            [projB, ~] = SumOfMatricize(B, 2*(targetDomain - 1)+1);
            result = U{targetDomain}*projB*V{targetDomain}';
            [~, maxIndex] = max(result');
            predictResult = maxIndex;
            for i = 1: CVFoldSize
                if(predictResult(validateIndex(i)) == label{targetDomain}(validateIndex(i)))
                    validateScore = validateScore + 1;
                end
            end
            validateIndex = validateIndex + CVFoldSize;
        end
        accuracy = validateScore/ numSampleInstance;
        avgObjectiveScore = sum(foldObjectiveScores)/ numCVFold;
        
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
        resultCellArray{t}{1} = avgObjectiveScore;
        resultCellArray{t}{2} = accuracy*100;
    end
    for numResult = 1:randomTryTime
        fprintf(resultFile, '%f,%f\n', resultCellArray{numResult}{1}, resultCellArray{numResult}{2});
    end
end
showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);
fprintf('done\n');
fclose(resultFile);
matlabpool close;