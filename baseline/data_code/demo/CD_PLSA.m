% function [Results, pz_d] = CD_PLSA(Train_Data,Test_Data,Parameter_Setting)
function CD_PLSA(datasetId)

% function [Results, pz_d] = CD_PLSA(Train_Data,Test_Data,Parameter_Setting)

% The common program for CD_PLSA, which can deal with multiple classes,
% multiple source domains and multiple target domains

%%%% Input:
% The parameter Train_data stores the file pathes of training data and the
% corresponding labels
% The parameter Test_data stores the file pathes of test data and the
% corresponding labels
% The parameter Parameterfile stores the parameter setting information

%%%% Output
% The variable Results is a matrix with size numIteration x numTarget, where
% numIteration is the number of iterations, numTarget is the number of
% target domains. Results record the detailed results of each iteration.

% The variable pz_d is a matrix with size n x c, where n is the number of
% instances in all target domains, specifically, n = n_1 + ... + nt (n_t is
% the number of instances in t-th target domain), c is the number of
% classes
% 
% Note that if you want to deal with large data set, you should set larget
% memory for Matlab. You can set it in the file C:\boot.ini (This may not
% be true in your system), change '/fastdetect' to '/fastdetect /3GB'.
% 
% Be good luck for your research, if you have any questions, you can
% contact the email: zhuangfz@ics.ict.ac.cn

numK = 64;
numC = 2;
numSource = 1;
numTarget = 1;
numIteration = 100;
numSampleFeature = 2000;
maxRandomTryTime = 10;

iscsvread = 1;

labelset = [];

if datasetId <= 6
    dataType = 1;
    prefix = '../../../../20-newsgroup/';
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../../../../Reuter/';
elseif datasetId >= 10 && datasetId  <= 13
    dataType = 2;
    prefix = '../../../../Animal_img/';
end

numDom = 2;
domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

% Load data from source and target domain data
X = createSparseMatrix_multiple(prefix, domainNameList, numDom, dataType);
for i = 1:numDom
    denseFeatureIndex = findDenseFeature(X{i}, numSampleFeature);
    X{i} = X{i}(:, denseFeatureIndex);
end
TrainY = load([prefix sprintf('source%d_label.csv', datasetId)]);
TestY = load([prefix sprintf('target%d_label.csv', datasetId)]);

TrainX = X{1};
TestX = X{2};

sampleSourceDataIndex = csvread(sprintf('../../../sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
sampleTargetDataIndex = csvread(sprintf('../../../sampleIndex/sampleTargetDataIndex%d.csv', datasetId));
numTrain = length(sampleSourceDataIndex);
numTest = length(sampleTargetDataIndex);
TrainX = TrainX(sampleSourceDataIndex, :);
TestX = TestX(sampleTargetDataIndex, :);
TrainY = TrainY(sampleSourceDataIndex, :);
TestY = TestY(sampleTargetDataIndex, :);
TrainX = TrainX';
TestX = TestX';
labelset = union(labelset, TrainY);
labelset = union(labelset, TestY);

numC = length(labelset);

avgAccuracy = 0;
for randomTryTime = 1:maxRandomTryTime
    pyz = rand(numK,numC);
%pyz = ones(numK,numC);
pyz = pyz/sum(sum(pyz));

start = 1;
if start == 1
    DataSetX = [TrainX TestX];
    Learn.Verbosity = 1;
    Learn.Max_Iterations = 200;
    Learn.heldout = .1; % for tempered EM only, percentage of held out data
    Learn.Min_Likelihood_Change = 1;
    Learn.Folding_Iterations = 20; % for TEM only: number of fiolding in iterations
    Learn.TEM = 0; %tempered or not tempered.
    size(DataSetX)
    [Pw_z,Pz_d,Pd,Li,perp,eta] = pLSA(DataSetX,[],numK,Learn); %start PLSA
    %xlswrite(strcat('pwz_','common_selected','.xls'),Pw_z);
end

%pwy = xlsread(strcat('pwz_','common_selected','.xls'));

pwy = Pw_z;

pw_yc_s = [];
for i = 1:numSource
    pw_yc_s = [pw_yc_s, pwy];
end
pw_yc_t = [];
for i = 1:numTarget
    pw_yc_t = [pw_yc_t, pwy];
end
clear pwy;

pd_zc_s = [];
for i = 1:numSource
    A = zeros(numTrain,numC);
    pos = 0;
    if i > 1
        for t = 1:i-1
            pos = pos + numTrain(t);
        end
    end
    if i == 1
        pos = 0;
    end
    for j = 1:numTrain
        for k = 1:numC
            if TrainY(pos+j) == labelset(k)
                A(j,k) = 1;
            end
        end
    end
    for k = 1:numC
        A(:,k) = A(:,k)/sum(A(:,k));
    end
    pd_zc_s = [pd_zc_s;A];
end

% random initialization for pd_z_t
% In our paper, pd_z_t is assigned as the predicted results by supervised classifiers 
pd_zc_t = [];
for i = 1:numTarget
    A = ones(numTest,numC);
    for k = 1:numC
        A(:,k) = A(:,k)/sum(A(:,k));
    end
    pd_zc_t = [pd_zc_t;A];
end

pc = zeros(1,numSource+numTarget);
numAll = sum(numTrain) + sum(numTest);
for i = 1:numSource
    pc(i) = numTrain/numAll;
end
for i = 1:numTarget
    pc(i+numSource) = numTest/numAll;
end

iter_results = [];

O_s = TrainX;
O_t = TestX;

for iterID = 1:numIteration

    temp_pw_yc_s = [];
    temp_pw_yc_t = [];
    temp_pd_zc_s = [];
    temp_pd_zc_t = [];
    temp_pyz = zeros(size(pyz));
    temp_pc = zeros(size(pc));

    stepLen = 2;
    numStep = fix(size(TrainX,1)/stepLen);
    step = fix(size(TrainX,1)/numStep);

    for i = 1:numSource
        pos = 0;
        if i > 1
            for t = 1:i-1
                pos = pos + numTrain(t);
            end
        end
        if i == 1
            pos = 0;
        end

        A = [];
        D = zeros(numK,numTrain);
        for stepID = 1:numStep
            if stepID < numStep
                tempsum2_s = pw_yc_s((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK)*pyz*pd_zc_s(pos+1:pos+numTrain,:)';
                tempsum2_s = tempsum2_s*pc(i);
                [xs ys] = find(tempsum2_s == 0);
                for q = 1:size(xs,1)
                    tempsum2_s(xs(q,1),ys(q,1)) = 1;
                end
                B = O_s((stepID-1)*step+1:stepID*step,pos+1:pos+numTrain)./tempsum2_s;
                A = [A; pw_yc_s((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK).*(B*pd_zc_s(pos+1:pos+numTrain,:)*pyz')];
                D = D + pw_yc_s((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK)'*B;
                clear B;
            end
            if stepID == numStep
                tempsum2_s = pw_yc_s((stepID-1)*step+1:size(O_s,1),(i-1)*numK+1:i*numK)*pyz*pd_zc_s(pos+1:pos+numTrain,:)';
                tempsum2_s = tempsum2_s*pc(i);
                [xs ys] = find(tempsum2_s == 0);
                for q = 1:size(xs,1)
                    tempsum2_s(xs(q,1),ys(q,1)) = 1;
                end
                B = O_s((stepID-1)*step+1:size(O_s,1),pos+1:pos+numTrain)./tempsum2_s;
                A = [A; pw_yc_s((stepID-1)*step+1:size(O_s,1),(i-1)*numK+1:i*numK).*(B*pd_zc_s(pos+1:pos+numTrain,:)*pyz')];
                D = D + pw_yc_s((stepID-1)*step+1:size(O_s,1),(i-1)*numK+1:i*numK)'*B;
                clear B;
            end
        end

        A = A*pc(i);
        for l = 1:numK
            A(:,l) = A(:,l)/sum(A(:,l));
        end
        temp_pw_yc_s = [temp_pw_yc_s A];
        clear A;
        
        temp_pyz = temp_pyz + (pyz.*(D*pd_zc_s(pos+1:pos+numTrain,:)))*pc(i);
        clear D;
        temp_pc(i) = sum(sum(TrainX(:,pos+1:pos+numTrain)));
    end

    for i = 1:numTarget
        pos = 0;
        if i > 1
            for t = 1:i-1
                pos = pos + numTest(t);
            end
        end
        if i == 1
            pos = 0;
        end

        A = [];
        C = zeros(numTest,numK);
        D = zeros(numK,numTest);
        for stepID = 1:numStep
            if stepID < numStep
                tempsum2_t = pw_yc_t((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK)*pyz*pd_zc_t(pos+1:pos+numTest,:)';
                tempsum2_t = tempsum2_t*pc(numSource+i);
                [xs ys] = find(tempsum2_t == 0);
                for q = 1:size(xs,1)
                    tempsum2_t(xs(q,1),ys(q,1)) = 1;
                end
                B = O_t((stepID-1)*step+1:stepID*step,pos+1:pos+numTest)./tempsum2_t;
                A = [A; pw_yc_t((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK).*(B*pd_zc_t(pos+1:pos+numTest,:)*pyz')];
                C = C + B'*pw_yc_t((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK);
                D = D + pw_yc_t((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK)'*B;
                clear B;
            end
            if stepID == numStep
                tempsum2_t = pw_yc_t((stepID-1)*step+1:size(O_t,1),(i-1)*numK+1:i*numK)*pyz*pd_zc_t(pos+1:pos+numTest,:)';
                tempsum2_t = tempsum2_t*pc(numSource+i);
                [xs ys] = find(tempsum2_t == 0);
                for q = 1:size(xs,1)
                    tempsum2_t(xs(q,1),ys(q,1)) = 1;
                end
                B = O_t((stepID-1)*step+1:size(O_t,1),pos+1:pos+numTest)./tempsum2_t;
                A = [A; pw_yc_t((stepID-1)*step+1:size(O_t,1),(i-1)*numK+1:i*numK).*(B*pd_zc_t(pos+1:pos+numTest,:)*pyz')];
                C = C + B'*pw_yc_t((stepID-1)*step+1:size(O_t,1),(i-1)*numK+1:i*numK);
                D = D + pw_yc_t((stepID-1)*step+1:size(O_t,1),(i-1)*numK+1:i*numK)'*B;
                clear B;
            end
        end

        A = A*pc(numSource+i);
        for l = 1:numK
            A(:,l) = A(:,l)/sum(A(:,l));
        end
        temp_pw_yc_t = [temp_pw_yc_t A];
        clear A;

        A = pd_zc_t(pos+1:pos+numTest,:).*(C*pyz);
        clear C;
        A = A*pc(numSource+i);
        for k = 1:numC
            A(:,k) = A(:,k)/sum(A(:,k));
        end
        temp_pd_zc_t = [temp_pd_zc_t;A];

        temp_pyz = temp_pyz + (pyz.*(D*pd_zc_t(pos+1:pos+numTest,:)))*pc(numSource+i);
        clear D;
        temp_pc(numSource+i) = sum(sum(TestX(:,pos+1:pos+numTest)));
    end

    temp_pyz = temp_pyz/sum(sum(temp_pyz));
    temp_pc = temp_pc/sum(temp_pc);

    pw_yc_s = temp_pw_yc_s;
    pw_yc_t = temp_pw_yc_t;
    pd_zc_t = temp_pd_zc_t;
    pyz = temp_pyz;
    pc = temp_pc;

    pz_d = [];
    for i = 1:numTarget
        pos = 0;
        if i > 1
            for t = 1:i-1
                pos = pos + numTest(t);
            end
        end
        if i == 1
            pos = 0;
        end

        pzd = zeros(numTest,numC);
        for j = 1:size(pzd,1)
            for k = 1:size(pzd,2)
                pzd(j,k) = pd_zc_t(pos+j,k)*sum(pyz(:,k));
            end
        end
        pz_d = [pz_d; pzd];
        nCorrect = 0;
        for j = 1:size(pzd,1)
            [va vi] = max(pzd(j,:));
            if labelset(vi) == TestY(pos+j)
                nCorrect = nCorrect + 1;
            end
        end
%         [iterID nCorrect/(numTest)]
        iter_results(iterID,i) = nCorrect/(numTest);
    end
    accuracy = nCorrect/(numTest);
    avgAccuracy = avgAccuracy + accuracy;
    for i = 1:size(pz_d,1)
         pz_d(i,:) = pz_d(i,:)/sum( pz_d(i,:));
    end

    O_s = TrainX;
    O_t = TestX;
end
disp(avgAccuracy/maxRandomTryTime);
Results = iter_results;
end