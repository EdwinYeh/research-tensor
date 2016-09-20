function [output]=solver(input,hyperparam)
% Input variable:
%   input.X{d} = instance x feature matrix in domain d
%   input.Y{d} = perception x instance matrix in domain d
%   input.S{d} = selector maxtrix of Y in domain d
%   input.Sxw{d} = Laplacian on numerator in domain d
%   input.Dxw{d} = Laplacian on denomerator in domain d
%   input.InstanceIsSeed{d} = instance x cluster matrix saving seed to update XW
%   hyperparam.beta = coef for 2-norm regularizer for W
%   hyperparam.gamma = 1-norm sparsity for Tensor projection
%   hyperparam.lambda = coef for Laplacian reg for instance
%   hyperparam.cpRank
%   hyperparam.perceptionClusterNum
% Output:
%   output.Tensor = Tensor
%   output.W = W
%   output.Objective = Objective


%  System parameter
%  Todo
%       add to hyperparam
tol = 10 ^-3;
debugMode = 1;


% Initialize :
%     W{d} : featureNum x clusterNum
%     A : perceptionClusterNum x cpRank
%     E{d} : instanceClusterNum x cpRank
%     XW{d} : instanceNum x instanceClusterNum
%     U{d}: perceptionNum x perceptionClusterNum
domainNum = length(input.Y);
instanceClusterNumList = zeros(1, domainNum);
instanceNumList = zeros(1, domainNum);
featureNumList = zeros(1, domainNum);
perceptionNumList = zeros(1, domainNum);

E=cell(domainNum,1);
W=cell(domainNum,1);
XW=cell(domainNum,1);
reconstructY=cell(length(input.Y),1);

for domId = 1:domainNum
    instanceClusterNumList(domId) = size(input.XW{domId}, 2);
    instanceNumList(domId) = size(input.X{domId}, 1);
    featureNumList(domId) = size(input.X{domId}, 2);
    perceptionNumList(domId) = size(input.Y{domId}, 1);
    
    E{domId} = randi(10, instanceClusterNumList(domId), hyperparam.cpRank);
    W{domId} = randi(10, featureNumList(domId), instanceClusterNumList(domId));
    XW{domId} = randi(10, instanceNumList(domId), instanceClusterNumList(domId));
end
A = randi(10,perceptionNumList(1),hyperparam.cpRank);
% Package A,E matrices into structure named "Tensor"
Tensor.A = A; Tensor.E = E;

%  Optimization main body
objectiveTrack = [];
objectiveScore = Inf;
terminateFlag = 0;
findNan = 0;
iter = 0;
maxIter = 100;

while terminateFlag<1 && ~findNan && iter < maxIter
    iter = iter + 1;
    for domID = 1:length(input.Y)
        Tensor = updateA(input, XW, Tensor,hyperparam);
%         disp('A:');
%         tmpScore = getObjectiveScore(input, U, XW, Tensor, hyperparam);
% %         save('debug.mat');
%         if isnan(tmpScore)            
%             findNan = 1;
%             break;
%         end
        
        Tensor = updateE(input, XW, Tensor, hyperparam, domID);
%         disp('E:');
%         tmpScore = getObjectiveScore(input, U, XW, Tensor, hyperparam);
% %         save('debug.mat');
%         if isnan(tmpScore)            
%             findNan = 1;
%             break;
%         end
        
        XW = updateXW(XW, input, Tensor, domID, hyperparam);
%         disp('XW:');
%         tmpScore = getObjectiveScore(input, U, XW, Tensor, hyperparam);
% %         save('debug.mat');
%         if isnan(tmpScore)            
%             findNan = 1;
%             break;
%         end
    end
    NewObjectiveScore = getObjectiveScore(input, XW, Tensor, hyperparam);
    if isnan(NewObjectiveScore)
        break;
    end
%     disp(NewObjectiveScore);
    %     Terminate Check
    relativeError = objectiveScore - NewObjectiveScore;
    objectiveScore = NewObjectiveScore;
    objectiveTrack(end+1) = NewObjectiveScore;
    terminateFlag = terminateFlag + terminateCheck(relativeError,tol);
end

% for domId = 1:domainNum
%     W = updateW(input,W,XW,domainNum);
% end

if debugMode
%     plot(objectiveTrack)
%     saveas(gcf,['ObjectiveTrack.png']);
end
% save('obj.mat','objectiveTrack');
output.objective = objectiveScore;
output.Tensor = Tensor;
output.W = W;
output.XW = XW;
for domId = 1:length(input.Y)
    reconstructY{domId} = getReconstructY(XW, Tensor, domId);
end
output.reconstructY = reconstructY;


function flag = terminateCheck(relativeError,tol)
flag = relativeError < tol;

function psi = getPsi(Tensor,domainIdx)
[M,n]=Khatrirao(Tensor,domainIdx);
cpRank=size(Tensor.E{1},2);
psi=zeros(cpRank);
for i = 1:n
    psi = psi + diag(M(i,:));
end

function [M,ProductionOfLabelNum]=Khatrirao(Tensor,domainIdx)
% Khatirao product on all the E matrices in Tensor, except E{domainIdx}
%  Todo : optimize
%           1.the preparation for large Label set (Total #Label cross domain) overflow
%           2.or sparse matrix calculation
%
% Prepare the matrices
cpRank=size(Tensor.E{1},2);
domainNum=length(Tensor.E);
ProductionOfLabelNum = 1;
for DomIdx = 1:domainNum
    %     ignore E{domainIdx}
    if DomIdx == domainIdx
        continue;
    end
    ProductionOfLabelNum = ProductionOfLabelNum * size(Tensor.E{DomIdx},1);
end
M=sparse(ProductionOfLabelNum ,cpRank);

for r = 1:cpRank
    M(:,r);
    E_col=1;
    % Note that E_col is a column
    for  DomIdx = 1:domainNum
        %     ignore E{domainIdx}
        if DomIdx == domainIdx
            continue;
        end
        tmpMatrix = Tensor.E{DomIdx};
        E_col=kron(E_col,tmpMatrix(:,r));
    end
    M(:,r)=M(:,r)+E_col;
end

function Y=getReconstructY(XW, Tensor, domId)
% Note that
%       Y hasn't been filtered/selected here
%       Y = X * W * proj, where proj = A*psi*E'
proj = projection(Tensor,domId);
Y=proj*XW{domId}';

function proj = projection(Tensor,domainIdx)
psi = getPsi(Tensor,domainIdx);
proj = Tensor.A * psi * Tensor.E{domainIdx}';

function LatinAlphabat = getlatin(M)
% Output:
%       LatinAlphabat : #col of M  x  #col of M
LatinAlphabat = diag(sum(M,1));

function sparsityTerm = get1normSparsityTerm(Tensor,domainIdx)
% If doamainIdx = 0, then times all the E{d}.
%  Output:
%   sparsityTerm: cpRank x cpRank
cpRank = size(Tensor.A,2);
sparsityTerm=ones(cpRank);
domainNum = length(Tensor.E);


sparsityTerm = sparsityTerm.*getlatin(Tensor.A);
for DomIdx = 1:domainNum
    if DomIdx == domainIdx
        continue;
    end
    sparsityTerm = sparsityTerm.*getlatin(Tensor.E{DomIdx});
end

function Tensor = updateA(input, XW, Tensor,hyperparam)
A=Tensor.A;
[r,c]=size(A);
% Calculate Numerator and Denominator
Numerator = zeros(r,c);
Denominator = zeros(r,c);
domainNum = length(Tensor.E);
big1 = ones(size(A,1),size(A,2));
for domId = 1:domainNum
    
    Numerator = Numerator ...
        + (input.Y{domId}.*input.S{domId}) ...
        * XW{domId} ...
        * Tensor.E{domId}*getPsi(Tensor,domId);
    
    Denominator = Denominator ...
        + (getReconstructY(XW, Tensor, domId).*input.S{domId}) ...
        * XW{domId} ...
        * Tensor.E{domId}*getPsi(Tensor,domId) ...
        +hyperparam.gamma * big1 * get1normSparsityTerm(Tensor,0);
end
A=A.*sqrt(Numerator./Denominator);
Tensor.A = A;

function Tensor = updateE(input, XW, Tensor, hyperparam, domId)
E=Tensor.E{domId};
[r,c]=size(E);
% Calculate Numerator and Denominator
Numerator = zeros(r,c);
Denominator = zeros(r,c);
big1 = ones(size(E,1),size(E,2));

%     Numerator
Numerator = Numerator + ...
    XW{domId}' ...
    * (input.Y{domId}.*input.S{domId})' ...
    * Tensor.A*getPsi(Tensor,domId);
%     Denominator
Denominator = Denominator + ...
    XW{domId}' ...
    * (getReconstructY(XW, Tensor, domId).*input.S{domId})' ...
    * Tensor.A*getPsi(Tensor,domId)...
    + hyperparam.gamma * big1 * get1normSparsityTerm(Tensor,domId);

E=E.*sqrt(Numerator./Denominator);
Tensor.E{domId} = E;

function W = updateW(input,W,XW,domID)
%  Note that W is a cell sturcture
% W{domainIdx}=input.X{domainIdx}\(XW{domainIdx});
[WRowSize, WColSize] = size(W{domID});
X = input.X{domID};
cvx_begin quiet
    variable tmpW(WRowSize, WColSize)
    minimize(norm(XW{domID}-X*tmpW,'fro'))
cvx_end
W{domID} = tmpW;

function XW = updateXW(XW, input, Tensor, domId, hyperparam)
    TmpXW = XW{domId};
    projectionH = projection(Tensor,domId);
    
    Numerator = (input.Y{domId}.*input.S{domId})'...
        * projectionH ...
        + hyperparam.lambda * input.Sxw{domId} * TmpXW;
    
    Denominator = (input.Y{domId}'.*input.S{domId}') ...
        * projectionH ...
        + hyperparam.lambda * input.Dxw{domId} * TmpXW;
    
    TmpXW=TmpXW.*sqrt(Numerator./Denominator);
    % force seed instance to be in the right cluster
    TmpXW(input.SeedSet{domId}, :) = input.SeedCluster{domId}(input.SeedSet{domId}, :);
    XW{domId} = TmpXW;
    

function objectiveScore = getObjectiveScore(input, XW, Tensor, hyperparam)
objectiveScore = 0;

for domId = 1:length(input.Y)
    objectiveScore = objectiveScore ...
        + norm((input.Y{domId}-projection(Tensor,domId)*XW{domId}').*input.S{domId},'fro')...
        + hyperparam.gamma*norm(projection(Tensor,domId),1) ...
        + hyperparam.lambda*trace(XW{domId}'*(input.Dxw{domId}-input.Sxw{domId})*XW{domId});
end

