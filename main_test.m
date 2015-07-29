clear;
createSparseTensor;
% if matlabpool('size') > 0 matlabpool close; end
% matlabpool('open', 'local', 4);
%delete('HSTLog.txt');
%diary('HSTLog.txt');
sourceLabel = load('20-newsgroup/train_label.csv');
targetLabel = load('20-newsgroup/test_label.csv');
Labels = cell(1, 2);

% configuration
isUpdateAE = true;
isUpdateFi = false;
isBinary = false;
%numTime = 20;
maxIter = 100;

numDom = 2;
predDomain = 2;
sourceDomain = 1;
numNews = [3954 3961];
numFeature = [58470 59474];
numSampleNews = 500;
numSampleFeature = 1500;
numTestSet = 100;
numNewsCluster = [3 3];
numFeatureCluster = [5 5];
lambda = 0.001;
gama = 0.001;
sigma = 1;
alpha = 0;
beta = 0;
delta = 0;

% disp(numSampleFeature);
%disp(sprintf('Configuration:\n\tisUpdateAE:%d\n\tisUpdateFi:%d\n\tisBinary:%d\n\tmaxIter:%d\n\t#domain:%d (predict domain:%d)', isUpdateAE, isUpdateFi, isBinary, maxIter, numDom, predDomain));
%disp(sprintf('#users:[%s]\n#items:[%s]\n#user_cluster:[%s]\n#item_cluster:[%s]', num2str(numNews(1:numDom)), num2str(numFeature(1:numDom)), num2str(numNewsCluster(1:numDom)), num2str(numFeatureCluster(1:numDom))));

%[groundTruthX, snapshot, idx] = preprocessing(numDom, predDomain);
%bestLambda = 0.1;
%bestAccuracy = 0;
str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numNewsCluster(i), numFeatureCluster(i));
end

str = str(1:length(str)-1);
%random initialize B
randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
randStr = round(randStr);
B = tensor(randStr);
Bcell = cell(1, numDom);
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

sampleTestIndex = randperm(numSampleNews, numTestSet);

for i = 1:numDom
    %for i = 1:numDom
    Bcell{i} = tensor();
    X{i} = importdata(sprintf('RealData/realX%d.mat', i));
    %randomly sample news
    SampleNewsIndex = randperm(numNews(i), numSampleNews);
    SampleNews = X{i}(SampleNewsIndex, :);
    if i == sourceDomain
        %Labels{i} = [sourceLabel(1:(numSampleNews/2), :); sourceLabel((numNews(i)-numSampleNews/2+1):numNews(i), :)];
        Labels{i} = sourceLabel(SampleNewsIndex, :);
    else
        %Labels{i} = [targetLabel(1:(numSampleNews/2), :); targetLabel((numNews(i)-numSampleNews/2+1):numNews(i), :)];
        Labels{i} = targetLabel(SampleNewsIndex, :);
    end
    denseFeatures = findDenseFeature(SampleNews, numSampleFeature);
    SampleX = SampleNews(:, denseFeatures);
    %     randItem = randperm(numFeature(i), numSampleFeature);
    %     SampleX = SampleNews(:, randItem);
    X{i} = SampleX;
    numNews(i) = numSampleNews;
    numFeature(i) = numSampleFeature;
    W{i} = zeros(numNews(i), numFeature(i));
    Y{i} = zeros(numNews(i), numNewsCluster(i));
    for j = 1: numNews(i)
        Y{i}(j, Labels{i}(j)) = 1;
    end
    if(i == sourceDomain)
        U{i} = Y{i};
        %i == predDomain
    else
        %define test set
        U{i} = rand(numNews(i),numNewsCluster(i));
        U{i}(:, numNewsCluster(i)) = 0;
        U{i} = fixTrainingSet(U{i}, Labels{i}, sampleTestIndex, -1);
    end
    V{i} = rand(numFeature(i),numFeatureCluster(i));
    V{i}(:, numFeatureCluster(i)) = 0;
    Su{i} = zeros(numNews(i), numNews(i));
    Du{i} = zeros(numNews(i), numNews(i));
    Lu{i} = zeros(numNews(i), numNews(i));
    Sv{i} = zeros(numFeature(i), numFeature(i));
    Dv{i} = zeros(numFeature(i), numFeature(i));
    Lv{i} = zeros(numFeature(i), numFeature(i));
    
    W{i}(X{i}~=0) = 1;
    
    %user
    disp('Calculating Su, Du, Lu');
    for useri = 1:numNews(i)
        for userj = 1:numNews(i)
            %ndsparse does not support norm()
            %dif = norm((X{i}(useri, :) - X{1}(userj,:)));
            difVector = X{i}(useri, :) - X{i}(userj, :);
            %ndsparse to normal value
            innerProduct = [0];
            innerProduct(1) = difVector* difVector';
            %dif = innerProduct^(1/2);
            dif = innerProduct;
            Su{i}(useri, userj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for useri = 1:numNews(i)
        Du{i}(useri,useri) = sum(Su{i}(useri,:));
    end
    Lu{i} = Du{i} - Su{i};
    %item
    disp('Calculating Sv, Dv, Lv');
    for itemi = 1:numFeature(i)
        for itemj = 1:numFeature(i)
            %ndsparse does not support norm()
            %dif = norm((X{i}(:,itemi) - itemTime(:,itemj)));
            difVector = X{i}(:, itemi) - X{i}(:, itemj);
            %ndsparse to normal value
            innerProduct = [0];
            innerProduct(1) = difVector'* difVector;
            %dif = innerProduct^(1/2);
            dif = innerProduct;
            Sv{i}(itemi, itemj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for itemi = 1:numFeature(i)
        Dv{i}(itemi,itemi) = sum(Sv{i}(itemi,:));
    end
    Lv{i} = Dv{i} - Sv{i};
end

%Iterative update
newEmpError = Inf;
empError = Inf;
iter = 0;
diff = -1;
empErrors = zeros(1,maxIter);
MAES = zeros(1,maxIter);
RMSES = zeros(1,maxIter);
%&& diff < 0
disp('Start iteration...');
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
        
        %sumAppError = 0;
        %for tmp = 1:length(threeMatrixB)
        %    oriB = threeMatrixB(:,:,tmp);
        %    appB = A*diag(CP.lambda(:).*U3(tmp,:)')*E';
        %    sumAppError = sumAppError + norm((oriB - appB), 1);
        %end
        %approximateError = sumAppError
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
        if(i == predDomain)
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
            U{i} = fixTrainingSet(U{i}, Labels{i}, sampleTestIndex, -1);
        end
        
        %update fi
        [r, c] = size(U3);
        nextThreeB = zeros(numNewsCluster(i), numFeatureCluster(i), r);
        %divThreeB = zeros(numNewsCluster(i), numFeatureCluster(i), r);
        sumFi = zeros(c, c);
        CPLamda = CP.lambda(:);
        for idx = 1:r
            %for idx = 1:r
            fi{idx} = diag(CPLamda.*U3(idx,:)');
            %[rFi cFi] = size(fi{idx});
            %onesFi = ones(rFi, cFi);
            %if isUpdateFi
            %    fi{idx} = fi{idx}.*sqrt((A'*U{i}'*X{i}*V{i}*E + delta*onesFi)./(A'*U{i}'*U{i}*A*fi{idx}*E'*V{i}'*V{i}*E));
            %    fi{idx}(isnan(fi{idx})) = 0;
            %    fi{idx}(~isfinite(fi{idx})) = 0;
            %end
            
            sumFi = sumFi + fi{idx};
            %if ~isUpdateAE
            %    nextThreeB(:,:,idx) = A*fi{idx}*E';
            %end
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
            for idx = 1:r
                %for idx = 1:r
                nextThreeB(:,:,idx) = A*fi{idx}*E';
                nextB = nextThreeB(:,:,idx);
            end
        end
        B = InverseThreeToOriginalB(tensor(nextThreeB), 2*(i-1)+1, eval(sprintf('[%s]', str)));
    end
    %disp(sprintf('\tCalculate this iterator error'));
    for i = 1:numDom
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
    
    %diff = newEmpError - empError
    empErrors(iter) = newEmpError;
    diff = empError - newEmpError;
    
    %pred
    %[projB, threeTensorB] = SumOfMatricize(B, 2*(predDomain - 1)+1);
    %pred = U{predDomain}*projB*V{predDomain}';
    
    %test = groundTruthX - snapshot;
    %testW = 1-W{predDomain};
    %testIdx = find(testW~=0);
    %if isBinary
    %    test(test ~= 0) = 1;
    %end
    
    %idx = find(test);
    %idx = testIdx;
    %MAES(iter) = norm((pred(idx) - test(idx)), 1)/length(idx);
    %RMSES(iter) = sqrt(norm((pred(idx) - test(idx)))^2/length(idx));
    %disp(sprintf('\tMAE:%f, RMSE:%f', MAES(iter), RMSES(iter)));
    %disp('=================================================================================')
end

[maxValue, maxIndex] = max(U{predDomain}');
correctCount = 0;
for i = 1: numTestSet
    %for i = 1: numNews(predDomain)
    if(maxIndex(sampleTestIndex(i)) == Labels{predDomain}(sampleTestIndex(i)))
        correctCount = correctCount + 1;
    end
end
correctRate = correctCount/ numTestSet;
%if correctRate > bestAccuracy
%    bestLambda = lambda;
%    bestAccuracy = correctRate;
%end
disp(correctRate);
numSampleFeature = numSampleFeature + 1000;
% matlabpool close;