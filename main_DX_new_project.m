clc;
% if parpool('size') > 0
%     parpool close;
% end
% parpool('open', 'local', 4);

% configuration
exp_title = sprintf('../exp_result/DX_advanced_new_projection_rank5_%d',datasetId);
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isRandom = true;
numSampleInstance = [500, 500];
numSampleFeature = [2000, 2000];
maxIter = 500;
randomTryTime = 5;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
    numInstanceCluster = 10;
    numFeatureCluster = 10;
    numClass = 2;
    sigma = 0.1;
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
    numInstanceCluster = 10;
    numFeatureCluster = 10;
    numClass = 2;
    sigma = 0.1;
elseif datasetId >= 10 && datasetId <= 13
    dataType = 2;
    prefix = '../Animal_img/';
    numInstanceCluster = 10;
    numFeatureCluster = 10;
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

numCVFold = 5;
CVFoldSize = numInstance(targetDomain)/ numCVFold;

directoryName = sprintf('../exp_result/DX/%d/', datasetId);
mkdir(directoryName);

% resultFile = fopen(sprintf('%s.csv', exp_title), 'w');
% fprintf(resultFile, 'lambda, gama, objectiveScore, accuracy, convergeTime\n');

disp('Start training');
for tuneGama = 0:1
    gama = 0.000001 * 100 ^ tuneGama;
    for tuneLambda = 0:0
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
            convergeTimer = tic;
            cpRank = 5;
            CP1 = rand(numInstanceCluster, cpRank);
            CP2 = rand(numFeatureCluster, cpRank);
            CP3 = rand(numInstanceCluster, cpRank);
            CP4 = rand(numFeatureCluster, cpRank);
            while (diff >= 0.0001  && iter < maxIter)%(abs(oldObjectiveScore - newObjectiveScore) >= 0.1 && iter < maxIter)
                iter = iter + 1;
                oldObjectiveScore = newObjectiveScore;
%                 fprintf('\t#Iterator:%d\n', iter);
%                 disp(newObjectiveScore);
                newObjectiveScore = 0;
                for dom = 1:numDom
                    % Compute projection of tensor by domain
                    [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
                    projB = A*sumFi*E';
                    % disp(sprintf('\t\tupdate V...'));
                    % Update V
                    V{dom} = V{dom}.*sqrt((X{dom}'*U{dom}*projB + gama*Sv{dom}*V{dom})./(V{dom}*projB'*U{dom}'*U{dom}*projB + gama*Dv{dom}*V{dom}));
                    V{dom}(isnan(V{dom})) = 0;
                    V{dom}(~isfinite(V{dom})) = 0;
                    % col normalize
                    [r, ~] = size(V{dom});
                    for tmpI = 1:r
                        bot = sum(abs(V{dom}(tmpI,:)));
                        if bot == 0
                            bot = 1;
                        end
                        V{dom}(tmpI,:) = V{dom}(tmpI,:)/bot;
                    end
                    V{dom}(isnan(V{dom})) = 0;
                    V{dom}(~isfinite(V{dom})) = 0;
                    
                    %disp(sprintf('\t\tupdate U...'));
                    %update U
                    U{dom} = U{dom}.*sqrt((X{dom}*V{dom}*projB'+lambda*Su{dom}*U{dom})./(U{dom}*projB*V{dom}'*V{dom}*projB'+lambda*Du{dom}*U{dom}));
                    U{dom}(isnan(U{dom})) = 0;
                    U{dom}(~isfinite(U{dom})) = 0;
                    %col normalize
                    [r ,~] = size(U{dom});
                    for tmpI = 1:r
                        bot = sum(abs(U{dom}(tmpI,:)));
                        if bot == 0
                            bot = 1;
                        end
                        U{dom}(tmpI,:) = U{dom}(tmpI,:)/bot;
                    end
                    U{dom}(isnan(U{dom})) = 0;
                    U{dom}(~isfinite(U{dom})) = 0;
                    
                    %update fi
                    
                    if isUpdateAE
                        %disp(sprintf('\t\tupdate A...'));
                        [rA, cA] = size(A);
                        onesA = ones(rA, cA);
                        A = A.*sqrt((U{dom}'*X{dom}*V{dom}*E*sumFi + alpha*(onesA))./(U{dom}'*U{dom}*A*sumFi*E'*V{dom}'*V{dom}*E*sumFi));
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        %A = (spdiags (sum(abs(A),1)', 0, cA, cA)\A')';
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        
                        %disp(sprintf('\t\tupdate E...'));
                        [rE ,cE] = size(E);
                        onesE = ones(rE, cE);
                        E = E.*sqrt((V{dom}'*X{dom}'*U{dom}*A*sumFi + beta*(onesE))./(V{dom}'*V{dom}*E*sumFi*A'*U{dom}'*U{dom}*A*sumFi));
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        %E = (spdiags (sum(abs(E),1)', 0, cE, cE)\E')';
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        
                        if dom == sourceDomain
                            CP1 = A;
                            CP2 = E;
                        else
                            CP3 = A;
                            CP4 = E;
                        end
                    end
                end
                %disp(sprintf('\tCalculate this iterator error'));
                for dom = 1:numDom
                    %for i = 1:numDom
                    [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
                    projB = A*sumFi*E';
                    result = U{dom}*projB*V{dom}';
                    normEmp = norm((X{dom} - result), 'fro')*norm((X{dom} - result), 'fro');
                    smoothU = lambda*trace(U{dom}'*Lu{dom}*U{dom});
                    smoothV = gama*trace(V{dom}'*Lv{dom}*V{dom});
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
        save(sprintf('%sU_%f_%f.mat', directoryName, lambda, gama), 'bestU');
%         targetTestingDataIndex = 1:CVFoldSize;
%         numCorrectPredict = 0;
%         for cvFold = 1: numCVFold
%             targetTrainingDataIndex = setdiff(1:numInstance(targetDomain),targetTestingDataIndex);
%             trainingData = [bestU{sourceDomain}; bestU{targetDomain}(targetTrainingDataIndex,:)];
%             trainingLabel = [sampledLabel{sourceDomain}; sampledLabel{targetDomain}(targetTrainingDataIndex, :)];
%             svmModel = fitcsvm(trainingData, trainingLabel, 'KernelFunction', 'rbf');
%             predictLabel = predict(svmModel, bestU{targetDomain}(targetTestingDataIndex,:));
%             for dataIndex = 1: CVFoldSize
%                 if sampledLabel{targetDomain}(targetTestingDataIndex(dataIndex)) == predictLabel(dataIndex)
%                     numCorrectPredict = numCorrectPredict + 1;
%                 end
%             end
%             targetTestingDataIndex = targetTestingDataIndex + CVFoldSize;
%         end
%         accuracy = numCorrectPredict/ (CVFoldSize*numCVFold);
%         fprintf('Lambda:%f, Gama:%f, ObjectiveScore:%f, Accuracy:%f%%\n', lambda, gama, bestRandomInitialObjectiveScore, accuracy);
%         fprintf(resultFile, '%f,%f,%f,%f,%f\n', lambda, gama, bestRandomInitialObjectiveScore, accuracy, bestConvergeTime);
    end
end
%showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);
fprintf('done\n');
% fclose(resultFile);
% parpool close;