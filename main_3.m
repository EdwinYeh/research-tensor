clear;clc;
% if matlabpool('size') > 0
%     matlabpool close;
% end
% matlabpool('open', 'local', 4);
%delete('HSTLog.txt');
%diary('HSTLog.txt');
datasetId = 1;
if((datasetId == 7)||(datasetId == 8)||(datasetId == 9))
    prefix = 'Reuter/';
elseif ((datasetId == 1)||(datasetId == 2)||(datasetId == 3)||(datasetId == 4)||(datasetId == 5)||(datasetId == 6))
    prefix = '../20-newsgroup/';
else
    prefix = 'Animal_img/';  
end

% configuration
isUpdateAE = true;
isUpdateFi = false;
isBinary = false;
%numTime = 20;
maxIter = 100;

numDom = 2;
targetDomain = 2;
sourceDomain = 1;
domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

numSourceList = [3914 3907 3783 3954 3830 3823 1237 1016 897 24295 24295 24295 24295 24295 24295 24295 24294 24294 17422 17422];
numTargetList = [3926 3910 3336 3961 3387 3371 1207 1043 897 6180 6180 6180 6180 6180 6180 6180 6180 6180 13052 13052];
numSourceFeatureList = [57312 59470 60800 58470 60800 60800 4771 4415 4563 2688 4096 2000 2000 2000 2000 252 2978 4470 4470 4470];
numTargetFeatureList = [57914 59474 61188 59474 61188 61188 4771 4415 4563 2688 4096 2000 2000 2000 2000 252 2978 4470 4470 4470];

numInstance = [numSourceList(datasetId) numTargetList(datasetId)];
numFeature = [numSourceFeatureList(datasetId) numTargetFeatureList(datasetId)];
numSampleNews = [500 500];
numSampleFeature = 3000;
if datasetId <= 9
    numInstanceCluster = [3 3];
    numFeatureCluster = [5 5];
elseif (datasetId > 9) && (datasetId < 20)
    numInstanceCluster = [41 11];
    numFeatureCluster = [5 5];
else
    numInstanceCluster = [26 26];
    numFeatureCluster = [5 5];
end
sigma = 1;
alpha = 0;
beta = 0;
delta = 0;
numCVFold = 5;
CVFoldSize = numSampleNews(targetDomain)/ numCVFold;
resultFile = fopen(sprintf('result%d.txt', datasetId), 'w');
sourceLabel = load(sprintf('%ssource%d_label.csv', prefix, datasetId));
targetLabel = load(sprintf('%starget%d_label.csv', prefix, datasetId));

str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster(i), numFeatureCluster(i));
end

str = str(1:length(str)-1);
%random initialize B
randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
randStr = round(randStr);

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
Labels = cell(1, 2);
trueLabels = cell(1, 2);
trueLabels{sourceDomain} = sourceLabel;
trueLabels{targetDomain} = targetLabel;

X = createSparseMatrix_multiple(prefix, domainNameList, numDom, 1);
%sample & calculate Laplacian
for i = 1: numDom
    %randomly sample news
    sampleNewsIndex = randperm(numInstance(i), numSampleNews(i));
    sampleNews = X{i}(sampleNewsIndex, :);
    if i == sourceDomain
        Labels{i} = trueLabels{i}(sampleNewsIndex, :);
    else
        Labels{i} = trueLabels{i}(sampleNewsIndex, :);
    end
    denseFeatures = findDenseFeature(sampleNews, numSampleFeature);
    X{i} = sampleNews(:, denseFeatures);
    numInstance(i) = numSampleNews(i);
    numFeature(i) = numSampleFeature;
    W{i} = zeros(numInstance(i), numFeature(i));
    Y{i} = zeros(numInstance(i), numInstanceCluster(i));
    for j = 1: numInstance(i)
        Y{i}(j, Labels{i}(j)) = 1;
    end
    Su{i} = zeros(numInstance(i), numInstance(i));
    Du{i} = zeros(numInstance(i), numInstance(i));
    Lu{i} = zeros(numInstance(i), numInstance(i));
    Sv{i} = zeros(numFeature(i), numFeature(i));
    Dv{i} = zeros(numFeature(i), numFeature(i));
    Lv{i} = zeros(numFeature(i), numFeature(i));
    
    W{i}(X{i}~=0) = 1;
    
    %user
    %disp('Calculating Su, Du, Lu');
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
    %disp('Calculating Sv, Dv, Lv');
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

bestScore = 0;
%reinitialize B, U, V
for tuneGama = 0:6
    gama = 0.000001 * 10 ^ tuneGama;
    for tuneLambda = 0:6
        lambda = 0.000001 * 10 ^ tuneLambda;
        validateScore = 0;
        validateIndex = 1: CVFoldSize;
        %fprintf('Lambda: %f, Gama: %f\n', lambda, gama);
        for fold = 1:numCVFold
            %re-initialize
            B = tensor(randStr);
            for i = 1:numDom
                if(i == sourceDomain)
                    U{i} = Y{i};
                else
                    U{i} = rand(numInstance(i),numInstanceCluster(i));
                    U{i}(:, numInstanceCluster(i)) = 0;
                    U{i} = fixTrainingSet(U{i}, Labels{i}, validateIndex);
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
                        U{i} = fixTrainingSet(U{i}, Labels{i}, validateIndex);
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
                if(predictResult(validateIndex(i)) == Labels{targetDomain}(validateIndex(i)))
                    validateScore = validateScore + 1;
                end
            end
            for c = 1:CVFoldSize
                validateIndex(c) = validateIndex(c) + CVFoldSize;
            end
        end
        validateAccuracy = validateScore/ numSampleNews(targetDomain);
        if validateScore > bestScore
            bestScore = validateScore;
            bestLambda = lambda;
            bestGama = gama;
        end
        fprintf('Lambda:%f, Gama:%f, ValidateAccuracy:%f, TheBest: %f\n', lambda, gama, validateAccuracy* 100, bestScore/ numSampleNews(targetDomain)* 100);
    end
end
fprintf(resultFile, sprintf('datasetId: %d\n', datasetId));
fprintf(resultFile, '(BestLambda,BestGama): (%f, %f)\n', bestLambda, bestGama);
fprintf(resultFile, 'BestScore: %f%%', bestScore/ numSampleNews(targetDomain)* 100);
fprintf('done\n');
fclose(resultFile);
matlabpool close;