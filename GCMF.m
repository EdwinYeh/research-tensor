function GCMF(exp_title, datasetId, numSampleInstance, numSampleFeature)
    clc;
    % if matlabpool('size') > 0
    %     matlabpool close;
    % end
    % matlabpool('open', 'local', 4);

    % configuration
    isSampleInstance = true;
    isSampleFeature = true;
    isURandom = true;
    %numTime = 20;
    maxIter = 100;
    randomTryTime = 10;

    prefix = '../20-newsgroup/';
    numDom = 2;
    sourceDomain = 1;
    targetDomain = 2;

    domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};
    allLabel = cell(1, numDom);

    numSourceInstanceList = [3913 3907 3783 3954 3830 3823 1237 1016 897 5000 5000 5000 5000 5000 5000 5000];
    numTargetInstanceList = [3925 3910 3336 3961 3387 3371 1207 1043 897 5000 5000 5000 5000 5000 5000 5000];
    numSourceFeatureList = [57312 59470 60800 58470 60800 60800 4771 4415 4563 10940 2688 2000 252 2000 2000 2000];
    numTargetFeatureList = [57914 59474 61188 59474 61188 61188 4771 4415 4563 10940];

    numInstance = [numSourceInstanceList(datasetId) numTargetInstanceList(datasetId)];
    numFeature = [numSourceFeatureList(datasetId) numTargetFeatureList(datasetId)];
    numInstanceCluster = [2 2];
    numFeatureCluster = [4 4];

    sigma = 1;
    numCVFold = 5;
    CVFoldSize = numSampleInstance/ numCVFold;
    resultFile = fopen(sprintf('result_%s.txt', exp_title), 'w');
    resultFile2 = fopen(sprintf('plot_data_%s.csv', exp_title), 'w');

    showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);

    % disp(numSampleFeature);
    %disp(sprintf('Configuration:\n\tisUpdateAE:%d\n\tisUpdateFi:%d\n\tisBinary:%d\n\tmaxIter:%d\n\t#domain:%d (predict domain:%d)', isUpdateAE, isUpdateFi, isBinary, maxIter, numDom, targetDomain));
    %disp(sprintf('#users:[%s]\n#items:[%s]\n#user_cluster:[%s]\n#item_cluster:[%s]', num2str(numInstance(1:numDom)), num2str(numFeature(1:numDom)), num2str(numInstanceCluster(1:numDom)), num2str(numFeatureCluster(1:numDom))));

    %Bcell = cell(1, numDom);
    Y = cell(1, numDom);
    W = cell(1, numDom);
    uc = cell(1, numDom);
    Sv = cell(1, numDom);
    Dv = cell(1, numDom);
    Lv = cell(1, numDom);
    Su = cell(1, numDom);
    Du = cell(1, numDom);
    Lu = cell(1, numDom);
    label = cell(1, numDom);

    X = createSparseMatrix_multiple(prefix, domainNameList, numDom, 1);

    for i = 1:numDom
        domainName = domainNameList{i};
        allLabel{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
    end

    for i = 1: numDom
        if isSampleInstance == true
            sampleInstanceIndex = randperm(numInstance(i), numSampleInstance);
            X{i} = X{i}(sampleInstanceIndex, :);
            numInstance(i) = numSampleInstance;
            label{i} = allLabel{i}(sampleInstanceIndex, :);
            Y{i} = zeros(numInstance(i), numInstanceCluster(i));
            for j = 1: numInstance(i)
                Y{i}(j, label{i}(j)) = 1;
            end
        end
        if isSampleFeature == true
            denseFeatures = findDenseFeature(X{i}, numSampleFeature);
            X{i} = X{i}(:, denseFeatures);
            numFeature(i) = numSampleFeature;
        end
    end

    % disp('Train logistic regression');
    % logisticCoefficient = glmfit(X{1}, Labels{1} - 1, 'binomial');

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
                %                 difVector = X{i}(useri, :) - X{i}(userj, :);
                %                 %ndsparse to normal value
                %                 dif = [0];
                %                 dif(1) = difVector* difVector';
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
                %                 difVector = X{i}(:, itemi) - X{i}(:, itemj);
                %                 %ndsparse to normal value
                %                 dif = [0];
                %                 dif(1) = difVector'* difVector;
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
    globalBestScore = Inf;
    globalBestAccuracy = 0;
    for tuneGama = 0:0
        gama = 0.001 * 1000 ^ tuneGama;
        for tuneLambda = 0:0
            lambda = 0.001 * 1000 ^ tuneLambda;
            time = round(clock);
            fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
            fprintf('Use Lambda: %f, Gama: %f\n', lambda, gama);
            localBestScore = Inf;
            localBestAccuracy = 0;
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
                    diff = -1;
                    ObjectiveScores = zeros(1,maxIter);
                    MAES = zeros(1,maxIter);
                    RMSES = zeros(1,maxIter);
                    while (abs(diff) >= 0.0001  && iter < maxIter)%(abs(ObjectiveScore - newObjectiveScore) >= 0.1 && iter < maxIter)
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
                                U{i} = fixTrainingSet(U{i}, label{i}, validateIndex);
                            end

                            %update H
                            HChild = zeros(numInstanceCluster(i), numFeatureCluster(i));
                            HMother = zeros(numInstanceCluster(i), numFeatureCluster(i));
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
                        parfor i = 1:numDom
                            result = U{i}*H*V{i}';
                            normEmp = norm(W{i}.*(X{i} - result))*norm(W{i}.*(X{i} - result));
                            smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                            smoothV = gama*trace(V{i}'*Lv{i}*V{i});
                            loss = normEmp + smoothU + smoothV;
                            newObjectiveScore = newObjectiveScore + loss;
                            %disp(sprintf('\t\tdomain #%d => empTerm:%f, smoothU:%f, smoothV:%f ==> objective score:%f', i, normEmp, smoothU, smoothV, loss));
                        end
                        %disp(sprintf('\tEmperical Error:%f', newObjectiveScore));
                        ObjectiveScores(iter) = newObjectiveScore;
                        %fprintf('iter:%d, error = %f\n', iter, newObjectiveScore);
                        diff = ObjectiveScore - newObjectiveScore;
                    end
                    %calculate validationScore
                    [~, maxIndex] = max(U{targetDomain}');
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
                Accuracy = validateScore/ numSampleInstance;
                if newObjectiveScore < globalBestScore
                    globalBestScore = newObjectiveScore;
                    globalBestAccuracy = Accuracy;
                    bestLambda = lambda;
                    bestGama = gama;
                end
                if newObjectiveScore < localBestScore
                    localBestScore = newObjectiveScore;
                    localBestAccuracy = Accuracy;
                end
                fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f\n', t, newObjectiveScore, Accuracy);
                fprintf(resultFile2, '%f,%f\n', newObjectiveScore, Accuracy);
            end
            fprintf('LocalBestScore:%f, LocalBestAccuracy:%f%%\nGlobalBestScore:%f, GlobalBestAccuracy: %f%%\n\n', localBestScore, localBestAccuracy*100, globalBestScore, globalBestAccuracy*100);
        end
    end

    showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);
    fprintf(resultFile, '(BestLambda,BestGama): (%f, %f)\n', bestLambda, bestGama);
    fprintf(resultFile, 'BestScore: %f%%', globalBestAccuracy* 100);
    fprintf('done\n');
    fclose(resultFile);
    fclose(resultFile2);
    %matlabpool close;
end