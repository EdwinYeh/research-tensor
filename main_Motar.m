clear;
clc;
% if matlabpool('size') > 0
%     matlabpool close;
% end
% matlabpool('open', 'local', 4);

% configuration
exp_title = 'Motar_10';
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isRandom = true;
datasetId = 10;
numSampleInstance = [500, 500];
numSampleFeature = [2000, 2000];
maxIter = 100;
randomTryTime = 5;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
    numInstanceCluster = 2;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
    numInstanceCluster = 2;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
elseif datasetId == 10
    dataType = 2;
    prefix = '../Animal_img/';
    numInstanceCluster = 2;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
elseif datasetId == 11
    dataType = 2;
    prefix = '../song/';
    numInstanceCluster = 2;
    numFeatureCluster = 5;
    numClass = 2;
    sigma = 0.1;
end
numDom = 2;
%sourceDomain = 1;
targetDomain = 2;

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

numSourceInstanceList = [3913 3906 3782 3953 3829 3822 1237 1016 897 4460 60];
numTargetInstanceList = [3925 3909 3338 3960 3389 3373 1207 1043 897 4601 30];
numSourceFeatureList = [57309 59463 60800 58463 60800 60800 4771 4415 4563 4940 26];
numTargetFeatureList = [57913 59474 61188 59474 61188 61188 4771 4415 4563 4940 26];

numInstance = [numSourceInstanceList(datasetId) numTargetInstanceList(datasetId)];
numFeature = [numSourceFeatureList(datasetId) numTargetFeatureList(datasetId)];

alpha = 0;
beta = 0;
numCVFold = 5;
CVFoldSize = numInstance(targetDomain)/ numCVFold;

showExperimentInfo(exp_title, datasetId, prefix, numInstance, numFeature);

Y = cell(1, numDom);
W = cell(1, numDom);
uc = cell(1, numDom);
Sv = cell(1, numDom);
Dv = cell(1, numDom);
Lv = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);
allLabel = cell(1, numDom);
sampledLabel = cell(1, numDom);

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);

for i = 1:numDom
    domainName = domainNameList{i};
    allLabel{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
end

for i = 1: numDom
    X{i} = normr(X{i});
    %Randomly sample instances & the corresponding labels
    if isSampleInstance == true
        sampleInstanceIndex = randperm(numInstance(i), numSampleInstance(i));
        X{i} = X{i}(sampleInstanceIndex, :);
        numInstance(i) = numSampleInstance(i);
        sampledLabel{i} = allLabel{i}(sampleInstanceIndex, :);
        Y{i} = zeros(numInstance(i), numInstanceCluster);
        for j = 1: numInstance(i)
            Y{i}(j, sampledLabel{i}(j)) = 1;
        end
    end
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature(i));
        X{i} = X{i}(:, denseFeatures);
        numFeature(i) = numSampleFeature(i);
    end
end

parfor i = 1: numDom
    W{i} = zeros(numInstance(i), numFeature(i));
    Su{i} = zeros(numInstance(i), numInstance(i));
    Du{i} = zeros(numInstance(i), numInstance(i));
    Lu{i} = zeros(numInstance(i), numInstance(i));
    Sv{i} = zeros(numFeature(i), numFeature(i));
    Dv{i} = zeros(numFeature(i), numFeature(i));
    Lv{i} = zeros(numFeature(i), numFeature(i));

    W{i}(X{i}~=0) = 1;

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

str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster, numFeatureCluster);
end
str = str(1:length(str)-1);

disp('Start training')
%initialize B, U, V
initV = cell(randomTryTime, numDom);
initU = cell(randomTryTime, numDom);
initB = cell(randomTryTime);
if isURandom == true
    for t = 1: randomTryTime
        [initU(t,:),initB{t},initV(t,:)] = randomInitialize(X, lable, numInstance, numFeature, numInstanceCluster, numFeatureCluster, numDom, true);
    end
end

resultFile = fopen(sprintf('score_accuracy_%s.csv', exp_title), 'w');

for tuneGama = 0:6
    gama = 0.000001 * 10 ^ tuneGama;
    for tuneLambda = 0:6
        lambda = 0.000001 * 10 ^ tuneLambda;
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Use Lambda:%f, Gama:%f\n', lambda, gama);
        for t = 1: randomTryTime
            validateScore = 0;
            validateIndex = 1: CVFoldSize;
            foldObjectiveScores = zeros(1,numCVFold);
            for fold = 1:numCVFold
                %Iterative update
                U = initU(t, :);
                V = initV(t, :);
                B = initB{t};
                for i = 1:numDom
                    if i == targetDomain
                        U{i} = fixTrainingSet(U{i}, sampledLabel{i}, validateIndex);
                    else
                        U{i} = Y{i};
                    end
                end
                newObjectiveScore = Inf;
                iter = 0;
                diff = -1;
                MAES = zeros(1,maxIter);
                RMSES = zeros(1,maxIter);
                %fprintf('Fold:%d(%d~%d), Iterative update\n', fold, min(validateIndex), max(validateIndex));
                while (diff >= 0.0001  && iter < maxIter)%(abs(oldObjectiveScore - newObjectiveScore) >= 0.1 && iter < maxIter)
                    iter = iter + 1;
                    oldObjectiveScore = newObjectiveScore;
%                         fprintf('\t#Iterator:%d', iter);
%                         disp([newObjectiveScore, diff]);
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
                        if(i == targetDomain)
                            U{i} = U{i}.*sqrt((X{i}*V{i}*projB' + lambda*Su{i}*U{i})./(U{i}*projB*V{i}'*V{i}*projB' + lambda*Du{i}*U{i}));
                            U{i}(isnan(U{i})) = 0;
                            U{i}(~isfinite(U{i})) = 0;
                            [r c] = size(U{i});
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
                        end

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
                    parfor i = 1:numDom
                        %for i = 1:numDom
                        [projB, ~] = SumOfMatricize(B, 2*(i - 1)+1);
                        result = U{i}*projB*V{i}';
                        normEmp = norm((X{i} - result))*norm((X{i} - result));
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
                foldObjectiveScores(fold) = newObjectiveScore;
                %calculate validationScore
                [~, maxIndex] = max(U{targetDomain}');
                predictResult = maxIndex;
                for i = 1: CVFoldSize
                    if(predictResult(validateIndex(i)) == sampledLabel{targetDomain}(validateIndex(i)))
                        validateScore = validateScore + 1;
                    end
                end
                for c = 1:CVFoldSize
                    validateIndex(c) = validateIndex(c) + CVFoldSize;
                end
            end
            accuracy = validateScore/ numSampleInstance(1);
            avgObjectiveScore = sum(foldObjectiveScores)/ numCVFold;
            fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
            fprintf(resultFile, '%f,%f\n', avgObjectiveScore, accuracy);
        end
        fprintf('LocalBestScore:%f, LocalBestAccuracy:%f%%\nGlobalBestScore:%f, GlobalBestAccuracy:%f%%\n\n',localBestScore, localBestAccuracy*100, globalBestScore, globalBestAccuracy*100);
    end
end
showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);
fprintf('done\n');
fclose(resultFile);
% matlabpool close;