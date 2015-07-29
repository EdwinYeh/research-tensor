clear;clc;
% if matlabpool('size') > 0
%     matlabpool close;
% end
% matlabpool('open', 'local', 4);

% configuration
isUpdateAE = true;
isUpdateFi = false;
isBinary = false;
isSampleInstance = true;
isSampleFeature = true;
isURandom = true;
%numTime = 20;
maxIter = 100;

prefix = '../20-newsgroup/';
exp_title = 'Motar1_2000_random';
datasetId = 1;
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
numSampleInstance = [500 500];
numSampleFeature = [2000 2000];
numInstanceCluster = [3 3];
numFeatureCluster = [5 5];

sigma = 1;
alpha = 0;
beta = 0;
delta = 0;
numCVFold = 5;
CVFoldSize = numSampleInstance(targetDomain)/ numCVFold;
resultFile = fopen(sprintf('result_%s.txt', exp_title), 'w');

showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature, numFeatureCluster(1));

% disp(numSampleFeature);
%disp(sprintf('Configuration:\n\tisUpdateAE:%d\n\tisUpdateFi:%d\n\tisBinary:%d\n\tmaxIter:%d\n\t#domain:%d (predict domain:%d)', isUpdateAE, isUpdateFi, isBinary, maxIter, numDom, targetDomain));
%disp(sprintf('#users:[%s]\n#items:[%s]\n#user_cluster:[%s]\n#item_cluster:[%s]', num2str(numInstance(1:numDom)), num2str(numFeature(1:numDom)), num2str(numInstanceCluster(1:numDom)), num2str(numFeatureCluster(1:numDom))));

%[groundTruthX, snapshot, idx] = preprocessing(numDom, targetDomain);
%bestLambda = 0.1;
%bestAccuracy = 0;
str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster(i), numFeatureCluster(i));
end

str = str(1:length(str)-1);
%random initialize B
randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
randStr = round(randStr);
%Bcell = cell(1, numDom);
X = cell(1, numDom);
Y = cell(1, numDom);
W = cell(1, numDom);
V = cell(1, numDom);
U = cell(1, numDom);
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
        sampleInstanceIndex = randperm(numInstance(i), numSampleInstance(i));
        X{i} = X{i}(sampleInstanceIndex, :);
        numInstance(i) = numSampleInstance(i);
        label{i} = allLabel{i}(sampleInstanceIndex, :);
        Y{i} = zeros(numInstance(i), numInstanceCluster(i));
        for j = 1: numInstance(i)
            Y{i}(j, label{i}(j)) = 1;
        end
    end
    if isSampleFeature == true
        denseFeatures = findDenseFeature(X{i}, numSampleFeature(i));
        X{i} = X{i}(:, denseFeatures);
        numFeature(i) = numSampleFeature(i);
        %denseFeatures = randperm(57312, numSampleFeature(1));
    end
end

% disp('Train logistic regression');
% logisticCoefficient = glmfit(X{1}, label{1} - 1, 'binomial');

for i = 1: numDom   
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
            %dif = norm((X{i}(useri, :) - X{1}(userj,:)));
            difVector = X{i}(useri, :) - X{i}(userj, :);
            %ndsparse to normal value
            dif = [0];
            dif(1) = difVector* difVector';
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
            %dif = norm((X{i}(:,itemi) - itemTime(:,itemj)));
            difVector = X{i}(:, itemi) - X{i}(:, itemj);
            %ndsparse to normal value
            dif = [0];
            dif(1) = difVector'* difVector;
            Sv{i}(itemi, itemj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for itemi = 1:numFeature(i)
        Dv{i}(itemi,itemi) = sum(Sv{i}(itemi,:));
    end
    Lv{i} = Dv{i} - Sv{i};
end

disp('Start training')
bestScore = 0;
%reinitialize B, U, V
for tuneGama = 0:6
    gama = 0.001 * 10 ^ tuneGama;
    for tuneLambda = 0:6
        lambda = 0.001 * 10 ^ tuneLambda;
        validateScore = 0;
        validateIndex = 1: CVFoldSize;
        fprintf('Use Lambda:%f, Gama:%f\n', lambda, gama);
        for fold = 1:numCVFold
            %re-initialize
            B = tensor(randStr);
            for i = 1:numDom
                if(i == targetDomain)
%                   disp('Assign U{target} with predict result of logistic regression')
                    if isURandom == true
                        U{i} = rand(numInstance(i), numInstanceCluster(i));
                        U{i}(:, numInstanceCluster(i)) = 0;
                    else
                        logisticPredictResult = glmval(logisticCoefficient, X{i}, 'probit');
                        U{i} = assignPredictResult(U{i}, logisticPredictResult, 1);
                    end
                    U{i} = fixTrainingSet(U{i}, label{i}, validateIndex);
                else
                    U{i} = Y{i};
                end
                V{i} = rand(numFeature(i),numFeatureCluster(i));
                V{i}(:, numFeatureCluster(i)) = 0;
            end
            %Iterative update
            newEmpError = Inf;
            empError = Inf;
            iter = 0;
            diff = -1;
            empErrors = zeros(1,maxIter);
            MAES = zeros(1,maxIter);
            RMSES = zeros(1,maxIter);
            %fprintf('Fold:%d(%d~%d), Iterative update\n', fold, min(validateIndex), max(validateIndex));
            while (abs(diff) >= 0.0001  && iter < maxIter)%(abs(empError - newEmpError) >= 0.1 && iter < maxIter)
                iter = iter + 1;
                empError = newEmpError;
                %disp(sprintf('\t#Iterator:%d', iter));
                %disp(newEmpError);
                newEmpError = 0;
                for i = 1:numDom
                    %disp(sprintf('\tdomain #%d update...', i));
                    [projB, threeMatrixB] = SumOfMatricize(B, 2*(i - 1)+1);
                    %bestCPR = FindBestRank(threeMatrixB, 50)
                    bestCPR = 2;
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
                    [r c] = size(V{i});
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
                        %U{i} = fixTrainingSet(U{i}, label{i}, validateIndex);
                    end
                    
                    %update fi
                    [r, c] = size(U3);
                    nextThreeB = zeros(numInstanceCluster(i), numFeatureCluster(i), r);
                    sumFi = zeros(c, c);
                    CPLamda = CP.lambda(:);
                    parfor idx = 1:r
                        %for idx = 1:r
                        fi{idx} = diag(CPLamda.*U3(idx,:)');
                        sumFi = sumFi + fi{idx};
                    end
                    if isUpdateAE
                        %disp(sprintf('\t\tupdate A...'));
                        [rA cA] = size(A);
                        onesA = ones(rA, cA);
                        A = A.*sqrt((U{i}'*X{i}*V{i}*E*sumFi + alpha*(onesA))./(U{i}'*U{i}*A*sumFi*E'*V{i}'*V{i}*E*sumFi));
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        %A = (spdiags (sum(abs(A),1)', 0, cA, cA)\A')';
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        
                        %disp(sprintf('\t\tupdate E...'));
                        [rE cE] = size(E);
                        onesE = ones(rE, cE);
                        E = E.*sqrt((V{i}'*X{i}'*U{i}*A*sumFi + beta*(onesE))./(V{i}'*V{i}*E*sumFi*A'*U{i}'*U{i}*A*sumFi));
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        %E = (spdiags (sum(abs(E),1)', 0, cE, cE)\E')';
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        
                        %disp(sprintf('\tcombine next iterator B...'));
                        parfor idx = 1:r
                            %for idx = 1:r
                            nextThreeB(:,:,idx) = A*fi{idx}*E';
                            nextB = nextThreeB(:,:,idx);
                        end
                    end
                    B = InverseThreeToOriginalB(tensor(nextThreeB), 2*(i-1)+1, eval(sprintf('[%s]', str)));
                end
                %disp(sprintf('\tCalculate this iterator error'));
                parfor i = 1:numDom
                    %for i = 1:numDom
                    [projB, threeTensorB] = SumOfMatricize(B, 2*(i - 1)+1);
                    result = U{i}*projB*V{i}';
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
            [maxValue, maxIndex] = max(U{targetDomain}');
            predictResult = maxIndex;
            for i = 1: CVFoldSize
                if(predictResult(validateIndex(i)) == label{targetDomain}(validateIndex(i)))
                    validateScore = validateScore + 1;
                end
            end
            for c = 1:CVFoldSize
                validateIndex(c) = validateIndex(c) + CVFoldSize;
            end
        end
        validateAccuracy = validateScore/ numSampleInstance(targetDomain);
        fprintf('Lambda:%f, Gama:%f, ValidateAccuracy:%f, TheBest: %f\n', lambda, gama, validateAccuracy, bestScore/ numSampleInstance(targetDomain)* 100);
        if validateScore > bestScore
            bestScore = validateScore;
            bestLambda = lambda;
            bestGama = gama;
        end
    end
end
showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature, numFeatureCluster(1));
fprintf(resultFile, '(BestLambda,BestGama): (%f, %f)\n', bestLambda, bestGama);
fprintf(resultFile, 'BestScore: %f%%', bestScore/ numSampleInstance(targetDomain)* 100);
fprintf('done\n');
fclose(resultFile);
% matlabpool close;