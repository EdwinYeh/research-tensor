clear
clc;
% if matlabpool('size') > 0
%     matlabpool close;
% end
% matlabpool('open', 'local', 4);

% configuration
exp_title = 'GCMF_7';
datasetId = 7;
numSampleInstance = [500, 500];
numSampleFeature = [2000, 2000];
isSampleInstance = true;
isSampleFeature = true;
isURandom = true;
%numTime = 20;
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
    sigma = 0.01;
end
numDom = 2;
sourceDomain = 1;
targetDomain = 2;

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};
allLabel = cell(1, numDom);

numSourceInstanceList = [3913 3906 3782 3953 3829 3822 1237 1016 897 4460 60];
numTargetInstanceList = [3925 3909 3338 3960 3389 3373 1207 1043 897 4601 30];
numSourceFeatureList = [57309 59463 60800 58463 60800 60800 4771 4415 4563 4940 39];
numTargetFeatureList = [57913 59474 61188 59474 61188 61188 4771 4415 4563 4940 39];

numInstance = [numSourceInstanceList(datasetId) numTargetInstanceList(datasetId)];
numFeature = [numSourceFeatureList(datasetId) numTargetFeatureList(datasetId)];

if isSampleInstance == true
    numInstance = numSampleInstance;
end

if isSampleFeature ==true
    numFeature = numSampleFeature;
end

numCVFold = 5;
CVFoldSize = numInstance(targetDomain)/ numCVFold;
resultFile = fopen(sprintf('score_accuracy_%s.csv', exp_title), 'w');

showExperimentInfo(exp_title, datasetId, prefix, numInstance, numFeature);

% disp(numSampleFeature);
%disp(sprintf('Configuration:\n\tisUpdateAE:%d\n\tisUpdateFi:%d\n\tisBinary:%d\n\tmaxIter:%d\n\t#domain:%d (predict domain:%d)', isUpdateAE, isUpdateFi, isBinary, maxIter, numDom, targetDomain));
%disp(sprintf('#users:[%s]\n#items:[%s]\n#user_cluster:[%s]\n#item_cluster:[%s]', num2str(numInstance(1:numDom)), num2str(numFeature(1:numDom)), num2str(numInstanceCluster(1:numDom)), num2str(numFeatureCluster(1:numDom))));

%Bcell = cell(1, numDom);
Y = cell(1, numDom);
uc = cell(1, numDom);
Sv = cell(1, numDom);
Dv = cell(1, numDom);
Lv = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);
label = cell(1, numDom);

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);

for i = 1: numDom
    domainName = domainNameList{i};
    allLabel{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
    label{i} = allLabel{i};
    X{i} = normr(X{i});
    if isSampleInstance == true
        sampleInstanceIndex = randperm(numInstance(i), numSampleInstance(i));
        X{i} = X{i}(sampleInstanceIndex, :);
        label{i} = allLabel{i}(sampleInstanceIndex, :);
    end
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature(i));
        %denseFeatures = randperm(numFeature(i), numSampleFeature);
        X{i} = X{i}(:, denseFeatures);
    end
    Y{i} = zeros(numInstance(i), numInstanceCluster);
    for j = 1: numInstance(i)
        Y{i}(j, label{i}(j)) = 1;
    end
end

% disp('Train logistic regression');
% logisticCoefficient = glmfit(X{1}, Labels{1} - 1, 'binomial');

parfor i = 1: numDom
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

disp('Start training')
%initialize B, U, V
initU = cell(randomTryTime, numDom);
initV = cell(randomTryTime, numDom);
initH = cell(randomTryTime);
if isURandom == true
    for t = 1: randomTryTime
        [initU(t,:),initH{t},initV(t,:)] = randomInitialize(numInstance, numFeature, numInstanceCluster, numFeatureCluster, numDom, false);
    end
end
for tuneGama = 0:6
    gama = 0.000001 * 10 ^ tuneGama;
    for tuneLambda = 0:6
        lambda = 0.000001 * 10 ^ tuneLambda;
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Use Lambda: %f, Gama: %f\n', lambda, gama);
        for t = 1: randomTryTime
            validateScore = 0;
            validateIndex = 1: CVFoldSize;
            for fold = 1:numCVFold
                %re-initialize
                U = initU(t, :);
                V = initV(t, :);
                H = initH{t};
                for i = 1:numDom
                    if i == targetDomain
                        U{i} = fixTrainingSet(U{i}, label{i}, validateIndex);
                    else
                        U{i} = Y{i};
                    end
                end
                HChildCell = cell(1, numDom);
                HMotherCell = cell(1, numDom);
                %Iterative update
                newObjectiveScore = Inf;
                ObjectiveScore = Inf;
                iter = 0;
                diff = Inf;
                foldObjectiveScores = zeros(1,numCVFold);
                MAES = zeros(1,maxIter);
                RMSES = zeros(1,maxIter);
                while (diff >= 0.0001  && iter < maxIter)%(abs(ObjectiveScore - newObjectiveScore) >= 0.1 && iter < maxIter)
                    iter = iter + 1;
                    ObjectiveScore = newObjectiveScore;
                    %disp(sprintf('\t#Iterator:%d', iter));
                    %disp(newObjectiveScore);
                    newObjectiveScore = 0;
                    for i = 1:numDom
                        %disp(sprintf('\t\tupdate V...'));
                        %update V
                        V{i} = V{i}.*sqrt((X{i}'*U{i}*H + gama*Sv{i}*V{i})./(V{i}*H'*U{i}'*U{i}*H + gama*Dv{i}*V{i}));
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
                            U{i} = U{i}.*sqrt((X{i}*V{i}*H' + lambda*Su{i}*U{i})./(U{i}*H*V{i}'*V{i}*H' + lambda*Du{i}*U{i}));
                            U{i}(isnan(U{i})) = 0;
                            U{i}(~isfinite(U{i})) = 0;
                            %col normalize
                            [r, ~] = size(U{i});
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

                        %update H
                        HChild = zeros(numInstanceCluster, numFeatureCluster);
                        HMother = zeros(numInstanceCluster, numFeatureCluster);
                        for j = 1:numDom
                            HChildCell{j} = U{j}'*X{j}*V{j};
                            HMotherCell{j} = U{j}'*U{j}*H*V{j}'*V{j};
                        end
                        for j = 1:numDom
                            HChild = HChild + HChildCell{j};
                            HMother = HMother + HMotherCell{j};
                        end
                        H = H.*sqrt(HChild./HMother);
                    end
                    %disp(sprintf('\tCalculate this iterator error'));
                    for i = 1:numDom
                        result = U{i}*H*V{i}';
                        normEmp = norm((X{i} - result))*norm((X{i} - result));
                        smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                        smoothV = gama*trace(V{i}'*Lv{i}*V{i});
                        loss = normEmp + smoothU + smoothV;
                        newObjectiveScore = newObjectiveScore + loss;
                        %disp(sprintf('\t\tdomain #%d => empTerm:%f, smoothU:%f, smoothV:%f ==> objective score:%f', i, normEmp, smoothU, smoothV, loss));
                    end
                    %disp(sprintf('\tEmperical Error:%f', newObjectiveScore));
                    foldObjectiveScores(fold) = newObjectiveScore;
                    fprintf('iter:%d, error = %f\n', iter, newObjectiveScore);
                    diff = ObjectiveScore - newObjectiveScore;
                end
                %calculate validationScore
                [~, maxIndex] = max(U{targetDomain}, [], 2);
                predictResult = maxIndex;
                for i = 1: CVFoldSize
                    if(predictResult(validateIndex(i)) == label{targetDomain}(validateIndex(i)))
                        validateScore = validateScore + 1;
                    end
                end
                for i = 1:CVFoldSize
                    validateIndex(i) = validateIndex(i) + CVFoldSize;
                end
            end
            avgObjectivescore = sum(foldObjectiveScores)/ numCVFold;
            accuracy = validateScore/ numInstance(targetDomain);
            fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t,avgObjectivescore, accuracy*100);
            fprintf(resultFile, '%f,%f\n', avgObjectivescore, accuracy);
        end
    end
end

showExperimentInfo(exp_title, datasetId, prefix, numInstance, numFeature);
fprintf('done\n');
fclose(resultFile);
%matlabpool close;
