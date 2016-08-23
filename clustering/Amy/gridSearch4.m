function [userResult] = gridSearch4(X, P, S, trueC, sigmas, alphas, betas)
% X: d*n data matrix
% P: perception matrix of the current user (its columns should be normalized to length = 1)
% S: n*k seed indicator
% trueC: n*k, groundtruth cluster assignment of the current user

numOfComb = length(sigmas)*length(alphas)*length(betas);
userResult = cell(numOfComb, 3);%[sigma, alpha, beta], resultF, resultObj
initIter = 2;
resultObj = zeros(initIter,1);
resultObj = resultObj + Inf;
resultF = cell(initIter,1);
i = 1;

for s=sigmas
    K = gaussian_kernel(X', s); % n*d
    for a=alphas
        for b=betas
            [s,a,b]
            resultObj = zeros(initIter,1);
            resultObj = resultObj + Inf;
            resultF = cell(initIter,1);
            for t=1:initIter
                [resultF{t,1},~,~,resultObj(t,1)] = solver4(K, P, S, a, b); % [F, C, i, objFinal]
            end
            idx = find(resultObj(:,1) == min(resultObj(:,1)));
            
            userResult{i,1} = [s,a,b];
            userResult{i,2} = resultF{idx(1),1};
            userResult{i,3} = resultObj(idx(1));
            
            i = i+1;
        end
    end
end
