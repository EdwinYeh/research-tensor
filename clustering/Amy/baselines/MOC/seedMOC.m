function [M] = seedMOC(data, cNum, seeds, minImpro, maxIter)
% data: #data*#feature (N*d) data matrix
% cNum: number of cluster
% seed: cNum*N seed indicator
% minImpro: if the difference between current and last objective function
% value <= minImpro, stop the iteration
% maxIter: the max iteration to run
%
% M: cNum*N cluster indicator matrix (remember to transport the final M to fit the dimension)
isUsingSeed = 0;
if (nargin < 3 || ~exist('seeds'))
    isUsingSeed = 0;
else
    isUsingSeed = 1;
end
if nargin < 4
    minImpro = 0.1;
    maxIter = 50;
end
if nargin < 5
    maxIter = 50;
end

[N,d] = size(data); % [#data,#feature]
M = randi([0,1],N,cNum); % cluster indicator, remember to transport the final M to fit the dimension
alpha = zeros(N,cNum); % alpha_ij = P(M_ij = 1)
A = rand(cNum,d); % A_ij: the degree of cluster i to generate feature j, The value is NOT restrict to [0,1], it could be any real number
lastObj = Inf; % the objective function value
lastObjDiff = Inf;

% control the cluster label of the seeds
if isUsingSeed == 1
    M = forceSeedClusterLabel(M,seeds);
end

for i=1:maxIter
    % update alpha
    for j=1:cNum
        alpha(:,j) = nnz(M(:,j))/N;
    end
    
    % update M
    newM = M;
    for j=1:N
        x = data(j,:);
        m = M(j,:);
        seedLabel = zeros(1,cNum);
        if isUsingSeed
            seedLabel(seeds(:,j)==1) = 1;
        end
        newM(j,:) = updateM(x,m,A,seedLabel);
    end
    
    % update A
    A = updateA(data,newM,A);
    
    M = newM;
    
    % calculate objective function value
    obj = calLoss(data,M,A) - sum(sum(log2(alpha)));
    impro = lastObj - obj;
    if abs(impro) <= minImpro
        %disp('break');
        break;
    end
    
    lastObjDiff = lastObj - obj;
    lastObj = obj;
end

M = M'; % return cNum*N matrix

end

function l = calLoss(X,M,A)
    Y = X-M*A;
    l = sum(sum(Y.*Y));
end

function M = forceSeedClusterLabel(M,seeds)
    [cNum,N] = size(seeds);
    seedIndex = find(seeds);
    for i=1:cNum
        M(seeds(i,:)==1,i) = 1;
    end
end

% function M = forceSeedClusterLabel(M,seeds)
%     seedIndex = find(seeds);
%     for j=seedIndex'
%         M(j,seeds(j)) = 1;
%     end
% end