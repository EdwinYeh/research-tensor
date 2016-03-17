clc;
% if parpool('size') > 0
%     parpool close;
% end
% parpool('open', 'local', 4);

% configuration
exp_title = sprintf('DX_advanced_%d',datasetId);
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isRandom = true;
numSampleInstance = [500, 500];
numSampleFeature = [2000, 2000];
maxIter = 500;
randomTryTime = 1;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
    numInstanceCluster = 5;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
    numInstanceCluster = 5;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
elseif datasetId >= 10 && datasetId <= 13
    dataType = 2;
    prefix = '../Animal_img/';
    numInstanceCluster = 5;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
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

for i = 1:numDom
    domainName = domainNameList{i};
    allLabel{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
end

for i = 1: numDom
    X{i} = normr(X{i});
    if i == sourceDomain
        sampleDataIndex = sampleSourceDataIndex;
    elseif i == targetDomain
        sampleDataIndex = sampleTargetDataIndex;
    end
    if isSampleInstance == true
        X{i} = X{i}(sampleDataIndex, :);
        numInstance(i) = numSampleInstance(i);
        sampledLabel{i} = allLabel{i}(sampleDataIndex, :);
    end
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature(i));
        X{i} = X{i}(:, denseFeatures);
        numFeature(i) = numSampleFeature(i);
    end
end

for i = 1: numDom
    Su{i} = zeros(numInstance(i), numInstance(i));
    Du{i} = zeros(numInstance(i), numInstance(i));
    Lu{i} = zeros(numInstance(i), numInstance(i));
    Sv{i} = zeros(numFeature(i), numFeature(i));
    Dv{i} = zeros(numFeature(i), numFeature(i));
    Lv{i} = zeros(numFeature(i), numFeature(i));
    
    %user
    fprintf('Domain%d: calculating Su, Du, Lu\n', i);
    for useri = 1:numInstance(i)
        for userj = 1:numInstance(i)
            %ndsparse does not support norm()
            dif = norm((X{i}(useri, :) - X{i}(userj,:)));
            Su{i}(useri, userj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for useri = 1:numInstance(i)
        Du{i}(useri,useri) = sum(Su{i}(useri,:));
    end
    Lu{i} = Du{i} - Su{i};
    %item
    fprintf('Domain%d: calculating Sv, Dv, Lv\n', i);
    for itemi = 1:numFeature(i)
        for itemj = 1:numFeature(i)
            %ndsparse does not support norm()
            dif = norm((X{i}(:,itemi) - X{i}(:,itemj)));
            Sv{i}(itemi, itemj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for itemi = 1:numFeature(i)
        Dv{i}(itemi,itemi) = sum(Sv{i}(itemi,:));
    end
    Lv{i} = Dv{i} - Sv{i};
end

%initialize B, U, V
initV = cell(randomTryTime, numDom);
initU = cell(randomTryTime, numDom);
initB = cell(randomTryTime);

str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster, numFeatureCluster);
end
str = str(1:length(str)-1);
for t = 1: randomTryTime
    for dom = 1: numDom
        initU{t,dom} = rand(numInstance(i), numInstanceCluster);
        initV{t,dom} = rand(numFeature(i), numFeatureCluster);
    end
    randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
    randStr = round(randStr);
    initB{t} = tensor(randStr);
end

numCVFold = 5;
CVFoldSize = numInstance(targetDomain)/ numCVFold;

resultFile = fopen(sprintf('result_%s.csv', exp_title), 'w');
fprintf(resultFile, 'lambda, gama, objectiveScore, accuracy, convergeTime\n');

disp('Start training');
for tuneGama = 0:3
    gama = 0.000001 * 100 ^ tuneGama;
    for tuneLambda = 0:3
        lambda = 0.000001 * 100 ^ tuneLambda;
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Use Lambda:%f, Gama:%f\n', lambda, gama);
        bestRandomInitialObjectiveScore = Inf;
        for t = 1: randomTryTime
            objectiveScore = 0;
            %Iterative update
            U = initU(t, :);
            V = initV(t, :);
            B = initB{t};
            newObjectiveScore = Inf;
            iter = 0;
            diff = Inf;
            %fprintf('Fold:%d(%d~%d), Iterative update\n', fold, min(validateIndex), max(validateIndex));
            convergeTimer = tic;
            while (diff >= 0.0001  && iter < maxIter)%(abs(oldObjectiveScore - newObjectiveScore) >= 0.1 && iter < maxIter)
                iter = iter + 1;
                oldObjectiveScore = newObjectiveScore;
%                 fprintf('\t#Iterator:%d\n', iter);
%                 disp(newObjectiveScore);
                newObjectiveScore = 0;
                for i = 1:numDom
                    %disp(sprintf('\tdomain #%d update...', i));
                    [projB, threeMatrixB] = SumOfMatricize(B, 2*(i - 1)+1);
                    %bestCPR = FindBestRank(threeMatrixB, 50)
                    bestCPR = 20;
                    CP = cp_apr(tensor(threeMatrixB), bestCPR, 'printitn', 0, 'alg', 'mu');%parafac_als(tensor(threeMatrixB), bestCPR);
                    A = CP.U{1};
                    E = CP.U{2};
                    U3 = CP.U{3};
                    
                    fi = cell(1, length(CP.U{3}));
                    
                    %disp(sprintf('\t\tupdate V...'));
                    %update V
                    V{i} = V{i}.*sqrt((X{i}'*U{i}*projB + gama*Sv{i}*V{i})./(V{i}*projB'*U{i}'*U{i}*projB + gama*Dv{i}*V{i}));
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
                    
                    %disp(sprintf('\t\tupdate U...'));
                    %update U
                    U{i} = U{i}.*sqrt((X{i}*V{i}*projB' + lambda*Su{i}*U{i})./(U{i}*projB*V{i}'*V{i}*projB' + lambda*Du{i}*U{i}));
                    U{i}(isnan(U{i})) = 0;
                    U{i}(~isfinite(U{i})) = 0;
                    %col normalize
                    [r ,~] = size(U{i});
                    for tmpI = 1:r
                        bot = sum(abs(U{i}(tmpI,:)));
                        if bot == 0
                            bot = 1;
                        end
                        U{i}(tmpI,:) = U{i}(tmpI,:)/bot;
                    end
                    U{i}(isnan(U{i})) = 0;
                    U{i}(~isfinite(U{i})) = 0;
                    
                    %update fi
                    [r, c] = size(U3);
                    nextThreeB = zeros(numInstanceCluster, numFeatureCluster, r);
                    sumFi = zeros(c, c);
                    CPLamda = CP.lambda(:);
                    parfor idx = 1:r
                        %for idx = 1:r
                        fi{idx} = diag(CPLamda.*U3(idx,:)');
                        sumFi = sumFi + fi{idx};
                    end
                    if isUpdateAE
                        %disp(sprintf('\t\tupdate A...'));
                        [rA, cA] = size(A);
                        onesA = ones(rA, cA);
                        A = A.*sqrt((U{i}'*X{i}*V{i}*E*sumFi + alpha*(onesA))./(U{i}'*U{i}*A*sumFi*E'*V{i}'*V{i}*E*sumFi));
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        %A = (spdiags (sum(abs(A),1)', 0, cA, cA)\A')';
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        
                        %disp(sprintf('\t\tupdate E...'));
                        [rE ,cE] = size(E);
                        onesE = ones(rE, cE);
                        E = E.*sqrt((V{i}'*X{i}'*U{i}*A*sumFi + beta*(onesE))./(V{i}'*V{i}*E*sumFi*A'*U{i}'*U{i}*A*sumFi));
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        %E = (spdiags (sum(abs(E),1)', 0, cE, cE)\E')';
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        
                        %disp(sprintf('\tcombine next iterator B...'));
                        parfor idx = 1:r
                            nextThreeB(:,:,idx) = A*fi{idx}*E';
                        end
                    end
                    B = InverseThreeToOriginalB(tensor(nextThreeB), 2*(i-1)+1, eval(sprintf('[%s]', str)));
                end
                %disp(sprintf('\tCalculate this iterator error'));
                for i = 1:numDom
                    %for i = 1:numDom
                    [projB, ~] = SumOfMatricize(B, 2*(i - 1)+1);
                    result = U{i}*projB*V{i}';
                    normEmp = norm((X{i} - result), 'fro')*norm((X{i} - result), 'fro');
                    smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                    smoothV = gama*trace(V{i}'*Lv{i}*V{i});
                    objectiveScore = normEmp + smoothU + smoothV;
                    newObjectiveScore = newObjectiveScore + objectiveScore;
                    %disp(sprintf('\t\tdomain #%d => empTerm:%f, smoothU:%f, smoothV:%f ==> objective score:%f', i, normEmp, smoothU, smoothV, objectiveScore));
                end
                %disp(sprintf('\tEmperical Error:%f', newObjectiveScore));
                %fprintf('iter:%d, error = %f\n', iter, newObjectiveScore);
                diff = oldObjectiveScore - newObjectiveScore;
            end
            convergeTime = toc(convergeTimer);
            if newObjectiveScore < bestRandomInitialObjectiveScore
                bestRandomInitialObjectiveScore = newObjectiveScore;
                bestU = U;
                bestConvergeTime = convergeTime;
            end
        end
        targetTestingDataIndex = 1:CVFoldSize;
        numCorrectPredict = 0;
        for cvFold = 1: numCVFold
            targetTrainingDataIndex = setdiff(1:numInstance(targetDomain),targetTestingDataIndex);
            trainingData = [bestU{sourceDomain}; bestU{targetDomain}(targetTrainingDataIndex,:)];
            trainingLabel = [sampledLabel{sourceDomain}; sampledLabel{targetDomain}(targetTrainingDataIndex, :)];
            svmModel = fitcsvm(trainingData, trainingLabel, 'KernelFunction', 'rbf');
            predictLabel = predict(svmModel, bestU{targetDomain}(targetTestingDataIndex,:));
            for dataIndex = 1: CVFoldSize
                if sampledLabel{targetDomain}(targetTestingDataIndex(dataIndex)) == predictLabel(dataIndex)
                    numCorrectPredict = numCorrectPredict + 1;
                end
            end
            targetTestingDataIndex = targetTestingDataIndex + CVFoldSize;
        end
        accuracy = numCorrectPredict/ (CVFoldSize*numCVFold);
        fprintf('Lambda:%f, Gama:%f, ObjectiveScore:%f, Accuracy:%f%%\n', lambda, gama, bestRandomInitialObjectiveScore, accuracy);
        fprintf(resultFile, '%f,%f,%f,%f,%f\n', lambda, gama, bestRandomInitialObjectiveScore, accuracy,bestConvergeTime);
    end
end
%showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);
fprintf('done\n');
fclose(resultFile);
% parpool close;