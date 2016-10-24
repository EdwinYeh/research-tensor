function [output]=solver_cp(input,hyperparam)

tol = 10 ^-6;

domainNum = length(input.Y);
instanceClusterNumList = zeros(1, domainNum);
instanceNumList = zeros(1, domainNum);
featureNumList = zeros(1, domainNum);
perceptionNumList = zeros(1, domainNum);

W=cell(domainNum,1);
XW=cell(domainNum,1);
reconstructY=cell(length(input.Y),1);

for domId = 1:domainNum
    instanceClusterNumList(domId) = size(input.XW{domId}, 2);
    instanceNumList(domId) = size(input.X{domId}, 1);
    featureNumList(domId) = size(input.X{domId}, 2);
    perceptionNumList(domId) = size(input.Y{domId}, 1);
    
    W{domId} = randi(10, featureNumList(domId), instanceClusterNumList(domId));
    XW{domId} = randi(10, instanceNumList(domId), instanceClusterNumList(domId));
end
threeMatrixB = randi(10, perceptionNumList(1), instanceClusterNumList(1), instanceClusterNumList(2));

%  Optimization main body
objectiveScore = Inf;
terminateFlag = 0;
findNan = 0;
iter = 0;
maxIter = 50;
TimeTracker = cell(maxIter ,  1);
ObjTracker = cell(maxIter, 1);
UTracker = cell(maxIter, 1);

timeBasis = 0;
while terminateFlag<5 && ~findNan && iter < maxIter
    iterTimer = tic;
    iter = iter + 1;
    for domID = 1:length(input.Y)
         projB = projectTensor(threeMatrixB, domID);
         bestCPR = 40;
         CP = cp_apr(tensor(threeMatrixB), bestCPR, 'printitn', 0, 'alg', 'mu');%parafac_als(tensor(threeMatrixB), bestCPR);
         A = CP.U{1};
         E = CP.U{2};
         U3 = CP.U{3};    
         fi = cell(1, length(CP.U{3}));
         CPLamda = CP.lambda(:);

         XW = updateXW(XW, input, projB, domID, hyperparam);

    threeMatrixB = updateAE(threeMatrixB, input, XW, A, E, U3, fi, projB, CPLamda, domID);
    end
    NewObjectiveScore = getObjectiveScore(input, XW, projB, hyperparam);
    if isnan(NewObjectiveScore)
        break;
    end
    %     Terminate Check
    relativeError = objectiveScore - NewObjectiveScore;
    objectiveScore = NewObjectiveScore;
    terminateFlag = terminateFlag + terminateCheck(relativeError,tol);
    timeBasis = timeBasis + toc(iterTimer);
    TimeTracker{iter} = timeBasis;
    ObjTracker{iter} = NewObjectiveScore;
    UTracker{iter} = XW;
end

output.objective = objectiveScore;
output.Tensor = threeMatrixB;
output.W = W;
output.XW = XW;
for domId = 1:length(input.Y)
    projB = projectTensor(threeMatrixB, domID);
    reconstructY{domId} = getReconstructY(XW, projB, domId);
end
output.reconstrucY = reconstructY;
output.Tracker{1} = TimeTracker;
output.Tracker{2} = ObjTracker;
output.Tracker{3} = UTracker;

function flag = terminateCheck(relativeError,tol)
flag = relativeError < tol;

function Y=getReconstructY(XW, projB, domId)
% Note that
%       Y hasn't been filtered/selected here
%       Y = X * W * proj, where proj = A*psi*E'
Y=projB*XW{domId}';

function nextThreeB = updateAE(threeMatrixB, input, U, A, E, U3, fi, projB, CPLamda, domId)
[r, c] = size(U3);
nextThreeB = zeros(size(threeMatrixB, 1), size(threeMatrixB, 2), size(threeMatrixB, 3));
sumFi = zeros(c, c);

for idx = 1:r
    fi{idx} = diag(CPLamda.*U3(idx,:)');
    sumFi = sumFi + fi{idx};
end
% Update A
[r,c]=size(A);
% Calculate Numerator and Denominator
Numerator = zeros(r,c);
Denominator = zeros(r,c);
domainNum = 2;
for domId2 = 1:domainNum
    
    Numerator = Numerator ...
        + (input.Y{domId2}.*input.S{domId2}) ...
        * U{domId2} ...
        * E*sumFi;
    
    Denominator = Denominator ...
        + (getReconstructY(U, projB, domId2).*input.S{domId2}) ...
        * U{domId2} ...
        * E*sumFi;
end
A=A.*sqrt(Numerator./Denominator);
% Update E
[r,c]=size(E);
% Calculate Numerator and Denominator
Numerator = zeros(r,c);
Denominator = zeros(r,c);

%     Numerator
Numerator = Numerator + ...
    U{domId}' ...
    * (input.Y{domId}.*input.S{domId})' ...
    * A*sumFi;
%     Denominator
Denominator = Denominator + ...
    U{domId}' ...
    * (getReconstructY(U, projB, domId).*input.S{domId})' ...
    * A*sumFi;

E=E.*sqrt(Numerator./Denominator);
for idx = 1:r
    nextThreeB(:,:,idx) = A*fi{idx}*E';
end

function XW = updateXW(XW, input, projB, domId, hyperparam)
    TmpXW = XW{domId};
    
    Numerator = (input.Y{domId}.*input.S{domId})'...
        * projB ...
        + hyperparam.lambda * input.Sxw{domId} * TmpXW;
    
    Denominator = TmpXW*TmpXW' ...
        * (input.Y{domId}'.*input.S{domId}') ...
        * projB ...
        + hyperparam.lambda * input.Dxw{domId} * TmpXW;
    
    TmpXW=TmpXW.*sqrt(Numerator./Denominator);
    % force seed instance to be in the right cluster
    TmpXW(input.SeedSet{domId}, :) = input.SeedCluster{domId}(input.SeedSet{domId}, :);
    XW{domId} = TmpXW;
    

function objectiveScore = getObjectiveScore(input, XW, projB, hyperparam)
objectiveScore = 0;

for domId = 1:length(input.Y)
    objectiveScore = objectiveScore ...
        + norm((input.Y{domId}-projB*XW{domId}').*input.S{domId},'fro')...
        + hyperparam.gamma*norm(projB,1) ...
        + hyperparam.lambda*trace(XW{domId}'*(input.Dxw{domId}-input.Sxw{domId})*XW{domId});
end
 
 function projB = projectTensor( threeMatrixB, mode )
     dim1 = size(threeMatrixB, 1);
     dim2 = size(threeMatrixB, 2);
     dim3 = size(threeMatrixB, 3);
     projB = zeros(dim1, dim2);
     if mode == 1
         for layer = 1:dim2
             for row = 1:dim1
                 for col = 1:dim3
                     projB(row, col) = projB(row, col) + threeMatrixB(row, layer, col);
                 end
             end
         end
     else
         for layer = 1:dim3
             for row = 1:dim1
                 for col = 1:dim2
                     projB(row, col) = projB(row, col) + threeMatrixB(row, layer, col);
                 end
             end
         end
     end