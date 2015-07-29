%clear all;clc;
%delete('HSTLog.txt');
%diary('HSTLog.txt');
createSparseTensor;
%if matlabpool('size') > 0 matlabpool close; end
%matlabpool('open', 'local', 4);


% configuration
isUpdateAE = true;
isUpdateFi = false;
isBinary = false;
numTime = 20;
maxIter = 100;

numDom = 2;
predDomain = 1;
%numUser = [20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20];
%numItem = [49 39 18];
%numItem = [49 36 45 37 50 35 40 43 42 32];
numUserCluster = [6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6];
numItemCluster = [6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6];
numOtherMode = 0;
numOtherCluster = [6 2 2 2];
lambda = 10;
gama = 1;
sigma = 1;
alpha = 0;
beta = 0;
delta = 0;

%disp(sprintf('Configuration:\n\tisUpdateAE:%d\n\tisUpdateFi:%d\n\tisBinary:%d\n\tmaxIter:%d\n\t#domain:%d (predict domain:%d)', isUpdateAE, isUpdateFi, isBinary, maxIter, numDom, predDomain));
%disp(sprintf('#users:[%s]\n#items:[%s]\n#user_cluster:[%s]\n#item_cluster:[%s]', num2str(numUser(1:numDom)), num2str(numItem(1:numDom)), num2str(numUserCluster(1:numDom)), num2str(numItemCluster(1:numDom))));

[groundTruthX, snapshot, idx] = preprocessing(numDom, predDomain);

str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numUserCluster(i), numItemCluster(i));
end
for i = 1:numOtherMode
    str = sprintf('%s%d,', str, numOtherCluster(i));
end
str = str(1:length(str)-1);

objs = zeros(1, numTime);
maes = zeros(1, numTime);
rmses = zeros(1, numTime);

for times = 1:numTime
%disp(sprintf('%d time', times));
%initial guess
%disp('Initializing...');
randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
randStr = round(randStr);
B = tensor(randStr);
Bcell = cell(1, numDom);
X = cell(1, numDom);
W = cell(1, numDom);
oriX = cell(1, numDom);
V = cell(1, numDom);
U = cell(1, numDom);
uc = cell(1, numDom);
Sv = cell(1, numDom);
Dv = cell(1, numDom);
Lv = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);
%disp('Load data and calculate similarity matrix...');
for i = 1:numDom
    %disp(sprintf('\tData#%d load', i));
    Bcell{i} = tensor();
    X{i} = importdata(sprintf('RealData/realX%dFinal.mat', i));
    if isBinary
        X{i}(X{i} ~= 0) = 1;
    end
    W{i} = zeros(numUser(i), numItem(i));
    oriX{i} = importdata(sprintf('RealData/realTensorX%d.mat', i));
    U{i} = rand(numUser(i),numUserCluster(i));
    U{i}(:, 1) = 0;
    V{i} = rand(numItem(i),numItemCluster(i));
    V{i}(:, 1) = 0;
    Su{i} = zeros(numUser(i), numUser(i));
    Du{i} = zeros(numUser(i), numUser(i));
    Lu{i} = zeros(numUser(i), numUser(i));
    Sv{i} = zeros(numItem(i), numItem(i));
    Dv{i} = zeros(numItem(i), numItem(i));
    Lv{i} = zeros(numItem(i), numItem(i));
    
    W{i}(X{i}~=0) = 1;
    
    %tune rank
    %bestR = FindBestRank(oriX{1}, 150)
    bestR = 20;
    CPX = cp_apr(oriX{i}, bestR, 'printitn', 0);%parafac_als(tensor(oriX{i}),bestR);
    userTime = CPX.U{1}*CPX.U{3}';
    itemTime = CPX.U{2}*CPX.U{3}';
    %csvwrite(sprintf('RealData/userTime%d.data', i), userTime);
    %csvwrite(sprintf('RealData/itemTime%d.data', i), itemTime);
    for useri = 1:numUser(i)
        for userj = 1:numUser(i)
            bot1 = norm(userTime(useri,:));
            bot2 = norm(userTime(userj,:));
            if bot1 == 0 bot1 = 1; end
            if bot2 == 0 bot2 = 1; end
            dif = norm((userTime(useri,:) - userTime(userj,:)));
            Su{i}(useri, userj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for useri = 1:numUser(i)
        Du{i}(useri,useri) = sum(Su{i}(useri,:));
    end
    Sui = Su{i};
    Dui = Du{i};
    Lu{i} = Du{i} - Su{i};
    for itemi = 1:numItem(i)
        for itemj = 1:numItem(i)
            bot1 = norm(itemTime(itemi,:));
            bot2 = norm(itemTime(itemj,:));
            if bot1 == 0 bot1 = 1; end
            if bot2 == 0 bot2 = 1; end
            dif = norm((itemTime(itemi,:) - itemTime(itemj,:)));
            Sv{i}(itemi, itemj) = exp(-(dif*dif)/(2*sigma));
        end
    end
    for itemi = 1:numItem(i)
        Dv{i}(itemi,itemi) = sum(Sv{i}(itemi,:));
    end
    Svi = Sv{i};
    Dvi = Dv{i};
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
%disp('Start iterator...');
while (abs(diff) >= 0.0001  && iter < maxIter)%(abs(empError - newEmpError) >= 0.1 && iter < maxIter)
    iter = iter + 1;
    %disp(sprintf('\t#Iterator:%d', iter));
    empError = newEmpError;
    newEmpError = 0;
    for i = 1:numDom
        %disp(sprintf('\tdomain #%d update...', i));
        [projB, threeMatrixB] = SumOfMatricize(B, 2*(i - 1)+1);
        %bestCPR = FindBestRank(threeMatrixB, 50)
        bestCPR = 20;
        CP = cp_apr(tensor(threeMatrixB), bestCPR, 'printitn', 0);%parafac_als(tensor(threeMatrixB), bestCPR);
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
        
        %update fi
        [r, c] = size(U3);
        nextThreeB = zeros(numUserCluster(i), numItemCluster(i), r);
        %divThreeB = zeros(numUserCluster(i), numItemCluster(i), r);
        sumFi = zeros(c, c);
        for idx = 1:r
            fi{idx} = diag(CP.lambda(:).*U3(idx,:)');
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
            parfor idx = 1:r
                nextThreeB(:,:,idx) = A*fi{idx}*E';
                nextB = nextThreeB(:,:,idx);
            end
        end
        B = InverseThreeToOriginalB(tensor(nextThreeB), 2*(i-1)+1, eval(sprintf('[%s]', str)));
    end
    %disp(sprintf('\tCalculate this iterator error'));
    for i = 1:numDom
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
    
    %pred
    [projB, threeTensorB] = SumOfMatricize(B, 2*(predDomain - 1)+1);
    pred = U{predDomain}*projB*V{predDomain}';

    test = groundTruthX - snapshot;
    testW = 1-W{predDomain};
    testIdx = find(testW~=0);
    if isBinary
        test(test ~= 0) = 1;
    end

    idx = find(test);
    %idx = testIdx;
    MAES(iter) = norm((pred(idx) - test(idx)), 1)/length(idx);
    RMSES(iter) = sqrt(norm((pred(idx) - test(idx)))^2/length(idx));
    %disp(sprintf('\tMAE:%f, RMSE:%f', MAES(iter), RMSES(iter)));
    %disp('=================================================================================')
end

[projB, threeTensorB] = SumOfMatricize(B, 2*(predDomain - 1)+1);
pred = U{predDomain}*projB*V{predDomain}';

test = groundTruthX - snapshot;
testW = 1-W{predDomain};
testIdx = find(testW~=0);
if isBinary
    test(test ~= 0) = 1;
end

idx = find(test);
%idx = testIdx;
%objective = newEmpError
%MAE = norm((pred(idx) - test(idx)), 1)/length(idx)
%plot(empErrors(1:80),MAES(1:80))
%hold on;
%f = plot(MAES);
%hold off;
%diary('off');
%myX3 = U{3}*projB*V{3}';
%[gr, gc] = size(groundTruthX3);
%MSE = norm(groundTruthX3 - myX3)*norm(groundTruthX3 - myX3)/(gr*gc)

bestIdx = find(empErrors==min(empErrors));
objs(times) = empErrors(bestIdx(1));
%disp(sprintf('times:#%d => objective score:%f', times, objs(times)));
maes(times) = MAES(bestIdx(1));
rmses(times) = RMSES(bestIdx(1));
%disp(sprintf('times:#%d => MAE:%f, RMSE:%f', times, maes(times), rmses(times)));

end

idx = find(objs==min(objs));
min_score_MAE = maes(idx);
min_score_RMSE = rmses(idx);

disp(sprintf('main : %f', min_score_MAE));

%matlabpool close;
%diary('off');