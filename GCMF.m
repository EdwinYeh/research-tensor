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

prefix = '../20-newsgroup/';
numDom = 2;
sourceDomain = 1;
targetDomain = 2;
randomTryTime = 5;
domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};
trueLabel = cell(1, numDom);

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
    trueLabel{i} = load([prefix, domainName(1:length(domainName)-4), '_label.csv']);
end

for i = 1: numDom
    if isSampleInstance == true
        sampleInstanceIndex = randperm(numInstance(i), numSampleInstance);
        X{i} = X{i}(sampleInstanceIndex, :);
        numInstance(i) = numSampleInstance;
        label{i} = trueLabel{i}(sampleInstanceIndex, :);
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
            dif = norm((X{i}(useri, :) - X{1}(userj,:)));
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
globalBestError = Inf;
globalBestAccuracy = 0;
if isURandom == true
    for t = 1: randomTryTime
        [initU(t,:),initH{t},initV(t,:)] = randomInitialize(Y, numInstance, numFeature, numInstanceCluster, numFeatureCluster, numDom, false);
    end
end
for tuneGama = 0:2
    gama = 0.01 * 10 ^ tuneGama;
    for tuneLambda = 0:2
        lambda = 100 * 10 ^ tuneLambda;
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Use Lambda: %f, Gama: %f\n', lambda, gama);
        localBestError = Inf;
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
                newEmpError = Inf;
                empError = Inf;
                iter = 0;
                diff = -1;
                empErrors = zeros(1,maxIter);
                MAES = zeros(1,maxIter);
                RMSES = zeros(1,maxIter);
                while (abs(diff) >= 0.0001  && iter < maxIter)%(abs(empError - newEmpError) >= 0.1 && iter < maxIter)
                    iter = iter + 1;
                    empError = newEmpError;
                    %disp(sprintf('\t#Iterator:%d', iter));
                    %disp(newEmpError);
                    newEmpError = 0;
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
                    for i = 1:numDom
                        result = U{i}*H*V{i}';
                        normEmp = norm(W{i}.*(X{i} - result))*norm(W{i}.*(X{i} - result));
                        smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                        smoothV = gama*trace(V{i}'*Lv{i}*V{i});
                        loss = normEmp + smoothU + smoothV;
                        newEmpError = newEmpError + loss;
                        %disp(sprintf('\t\tdomain #%d => empTerm:%f, smoothU:%f, smoothV:%f ==> objective score:%f', i, normEmp, smoothU, smoothV, loss));
                    end
                    %disp(sprintf('\tEmperical Error:%f', newEmpError));
                    empErrors(iter) = newEmpError;
                    %fprintf('iter:%d, error = %f\n', iter, newEmpError);
                    diff = empError - newEmpError;
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
            validateAccuracy = validateScore/ numSampleInstance;
            if validateAccuracy > globalBestAccuracy
                globalBestAccuracy = validateAccuracy;
                bestLambda = lambda;
                bestGama = gama;
            end
            if validateAccuracy > localBestAccuracy
                localBestAccuracy = validateAccuracy;
            end
            if newEmpError < globalBestError
                globalBestError = newEmpError;
            end
            if newEmpError < localBestError
                localBestError = newEmpError;
            end
            fprintf('Initial try: %d, ValidateError:%f, ValidateAccuracy:%f\n', t, newEmpError, validateAccuracy);
        end
        fprintf('LocalBestError:%f, LocalBestAccuracy:%f%%\nGlobalBestError:%f, GlobalBestAccuracy: %f%%\n\n', localBestError, localBestAccuracy*100, globalBestError, globalBestAccuracy*100);
    end
end

showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);
fprintf(resultFile, '(BestLambda,BestGama): (%f, %f)\n', bestLambda, bestGama);
fprintf(resultFile, 'BestScore: %f%%', globalBestAccuracy* 100);
fprintf('done\n');
fclose(resultFile);
%matlabpool close;