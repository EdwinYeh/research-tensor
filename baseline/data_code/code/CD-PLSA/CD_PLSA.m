function [Results, pz_d] = CD_PLSA(Train_Data,Test_Data,Parameter_Setting)

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




numK = 0;
numC = 0;
numSource = 0;

%read the parameters
fid=fopen(Parameter_Setting);
numK = str2num(fgetl(fid));
numIteration = str2num(fgetl(fid));
fclose(fid);

iscsvread = 0;
fid=fopen(Train_Data);
numSource = str2num(fgetl(fid));
fid1=fopen(fgetl(fid));
A = fgetl(fid1);
B = find(A == ',');
if length(B) == 2
    iscsvread = 1;
end
fclose(fid1);
fclose(fid);

labelset = [];

if iscsvread == 1
    % read source domain data
    fid=fopen(Train_Data);
    numSource = str2num(fgetl(fid));
    TrainX = [];
    TrainY = [];
    numTrain = [];
    for i = 1:numSource
        A = csvread(fgetl(fid));
        B = spconvert(A);
        TrainX = [TrainX B];
        C = textread(fgetl(fid));
        TrainY = [TrainY C'];
        numTrain(1,i) = length(C);
        labelset = union(labelset,C');
        clear A;
        clear B;
        clear C;
    end
    fclose(fid);

    % read target domain data
    fid=fopen(Test_Data);
    numTarget = str2num(fgetl(fid));
    TestX = [];
    TestY = [];
    numTest = [];
    for i = 1:numTarget
        A = csvread(fgetl(fid));
        B = spconvert(A);
        TestX = [TestX B];
        C = textread(fgetl(fid));
        TestY = [TestY C'];
        numTest(1,i) = length(C);
        labelset = union(labelset,C');
        clear A;
        clear B;
        clear C;
    end
    fclose(fid);
else
    % read source domain data
    fid=fopen(Train_Data);
    numSource = str2num(fgetl(fid));
    TrainX = [];
    TrainY = [];
    numTrain = [];
    for i = 1:numSource
        A = load(fgetl(fid));
        B = spconvert(A);
        TrainX = [TrainX B];
        C = textread(fgetl(fid));
        TrainY = [TrainY C'];
        numTrain(1,i) = length(C);
        labelset = union(labelset,C');
        clear A;
        clear B;
        clear C;
    end
    fclose(fid);

    % read target domain data
    fid=fopen(Test_Data);
    numTarget = str2num(fgetl(fid));
    TestX = [];
    TestY = [];
    numTest = [];
    for i = 1:numTarget
        A = load(fgetl(fid));
        B = spconvert(A);
        TestX = [TestX B];
        C = textread(fgetl(fid));
        TestY = [TestY C'];
        numTest(1,i) = length(C);
        labelset = union(labelset,C');
        clear A;
        clear B;
        clear C;
    end
    fclose(fid);
end

numC = length(labelset);
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
    Learn.TEM = 0; %tempered or not tempered
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
    A = zeros(numTrain(i),numC);
    pos = 0;
    if i > 1
        for t = 1:i-1
            pos = pos + numTrain(t);
        end
    end
    if i == 1
        pos = 0;
    end
    for j = 1:numTrain(i)
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
    A = ones(numTest(i),numC);
    for k = 1:numC
        A(:,k) = A(:,k)/sum(A(:,k));
    end
    pd_zc_t = [pd_zc_t;A];
end

pc = zeros(1,numSource+numTarget);
numAll = sum(numTrain) + sum(numTest);
for i = 1:numSource
    pc(i) = numTrain(i)/numAll;
end
for i = 1:numTarget
    pc(i+numSource) = numTest(i)/numAll;
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

    stepLen = 3000;
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
        D = zeros(numK,numTrain(i));
        for stepID = 1:numStep
            if stepID < numStep
                tempsum2_s = pw_yc_s((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK)*pyz*pd_zc_s(pos+1:pos+numTrain(i),:)';
                tempsum2_s = tempsum2_s*pc(i);
                [xs ys] = find(tempsum2_s == 0);
                for q = 1:size(xs,1)
                    tempsum2_s(xs(q,1),ys(q,1)) = 1;
                end
                B = O_s((stepID-1)*step+1:stepID*step,pos+1:pos+numTrain(i))./tempsum2_s;
                A = [A; pw_yc_s((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK).*(B*pd_zc_s(pos+1:pos+numTrain(i),:)*pyz')];
                D = D + pw_yc_s((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK)'*B;
                clear B;
            end
            if stepID == numStep
                tempsum2_s = pw_yc_s((stepID-1)*step+1:size(O_s,1),(i-1)*numK+1:i*numK)*pyz*pd_zc_s(pos+1:pos+numTrain(i),:)';
                tempsum2_s = tempsum2_s*pc(i);
                [xs ys] = find(tempsum2_s == 0);
                for q = 1:size(xs,1)
                    tempsum2_s(xs(q,1),ys(q,1)) = 1;
                end
                B = O_s((stepID-1)*step+1:size(O_s,1),pos+1:pos+numTrain(i))./tempsum2_s;
                A = [A; pw_yc_s((stepID-1)*step+1:size(O_s,1),(i-1)*numK+1:i*numK).*(B*pd_zc_s(pos+1:pos+numTrain(i),:)*pyz')];
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
        
        temp_pyz = temp_pyz + (pyz.*(D*pd_zc_s(pos+1:pos+numTrain(i),:)))*pc(i);
        clear D;
        temp_pc(i) = sum(sum(TrainX(:,pos+1:pos+numTrain(i))));
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
        C = zeros(numTest(i),numK);
        D = zeros(numK,numTest(i));
        for stepID = 1:numStep
            if stepID < numStep
                tempsum2_t = pw_yc_t((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK)*pyz*pd_zc_t(pos+1:pos+numTest(i),:)';
                tempsum2_t = tempsum2_t*pc(numSource+i);
                [xs ys] = find(tempsum2_t == 0);
                for q = 1:size(xs,1)
                    tempsum2_t(xs(q,1),ys(q,1)) = 1;
                end
                B = O_t((stepID-1)*step+1:stepID*step,pos+1:pos+numTest(i))./tempsum2_t;
                A = [A; pw_yc_t((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK).*(B*pd_zc_t(pos+1:pos+numTest(i),:)*pyz')];
                C = C + B'*pw_yc_t((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK);
                D = D + pw_yc_t((stepID-1)*step+1:stepID*step,(i-1)*numK+1:i*numK)'*B;
                clear B;
            end
            if stepID == numStep
                tempsum2_t = pw_yc_t((stepID-1)*step+1:size(O_t,1),(i-1)*numK+1:i*numK)*pyz*pd_zc_t(pos+1:pos+numTest(i),:)';
                tempsum2_t = tempsum2_t*pc(numSource+i);
                [xs ys] = find(tempsum2_t == 0);
                for q = 1:size(xs,1)
                    tempsum2_t(xs(q,1),ys(q,1)) = 1;
                end
                B = O_t((stepID-1)*step+1:size(O_t,1),pos+1:pos+numTest(i))./tempsum2_t;
                A = [A; pw_yc_t((stepID-1)*step+1:size(O_t,1),(i-1)*numK+1:i*numK).*(B*pd_zc_t(pos+1:pos+numTest(i),:)*pyz')];
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

        A = pd_zc_t(pos+1:pos+numTest(i),:).*(C*pyz);
        clear C;
        A = A*pc(numSource+i);
        for k = 1:numC
            A(:,k) = A(:,k)/sum(A(:,k));
        end
        temp_pd_zc_t = [temp_pd_zc_t;A];

        temp_pyz = temp_pyz + (pyz.*(D*pd_zc_t(pos+1:pos+numTest(i),:)))*pc(numSource+i);
        clear D;
        temp_pc(numSource+i) = sum(sum(TestX(:,pos+1:pos+numTest(i))));
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

        pzd = zeros(numTest(i),numC);
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
        [iterID nCorrect/(numTest(i))]
        iter_results(iterID,i) = nCorrect/(numTest(i));
    end
    for i = 1:size(pz_d,1)
         pz_d(i,:) = pz_d(i,:)/sum( pz_d(i,:));
    end

    O_s = TrainX;
    O_t = TestX;
end

Results = iter_results;
