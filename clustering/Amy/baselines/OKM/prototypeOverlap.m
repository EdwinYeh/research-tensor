function [ newmeans ] = prototypeOverlap(X, A ,means)

% means: c * dimesion
% A = nodes * c

    [n, c]= size(A);
    [~, d] = size(X);
    
    alpha = 1 ./ (sum(A,2).*sum(A,2));
    
    newmeans = zeros(size(means));
 
    for h=1:c
       idx = zeros(1, c);
       idx(1,h) = 1;
       mh = repmat(sum(A,2),1,d).* X - A(:,~idx)*means(~idx,:);
     
       tmp = alpha(logical(A(:,h)));
       newmeans(h,:) =  (1/sum(tmp))*sum(repmat(tmp, 1, d).*mh(logical(A(:,h)),:),1);
     
    end
    

end

